import os
from typing import Any, Optional

from eureka_ml_insights.core import (
    DataProcessing,
    Inference,
    PromptProcessing,
)
from eureka_ml_insights.core.eval_reporting import EvalReporting
from eureka_ml_insights.data_utils import ColumnRename, SequenceTransform, RunPythonTransform
from eureka_ml_insights.data_utils.data import (
    MMDataLoader,
    DataReader,
    HFDataReader,
)
from eureka_ml_insights.configs import(
    DataProcessingConfig,
    DataSetConfig,
    InferenceConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.configs import ExperimentConfig


class TEST_PIPELINE(ExperimentConfig):
    """This class specifies the config for running IFEval benchmark on any model"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, 
        **kwargs: dict[str, Any]) -> PipelineConfig:

        # data preprocessing
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": "/home/sayouse/git/test_input.jsonl",
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        # inference component
        ## The inference component is configured to run in chat mode,
        ## which means that it will start/maintain a history of previous messages
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {
                    "path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl"),
                },
            ),
            max_concurrent=2,
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
            chat_mode=True,
        )

        ## Prepare the prompt for the next chat turn, 
        ## in this case the prompt is a simple "Are you sure?" question
        self.post_processing_comp = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={"model_output": "first_model_output"}),
                            RunPythonTransform("df['prompt'] = 'Are you sure?'"),
                        ],
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "post_processing_output"),
        )
        ## Specify what extra columns the data loader should load, 
        ## which for chat mode should include the previous_messages columns
        self.second_inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {
                    "path": os.path.join(self.post_processing_comp.output_dir, "transformed_data.jsonl"),
                    "misc_columns": ["previous_messages"],
                },
            ),
            max_concurrent=2,
            output_dir=os.path.join(self.log_dir, "second_inference_result"),
            resume_from=kwargs.get("resume_from_2", None),
            chat_mode=True,
        )

        # Configure the pipeline
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.post_processing_comp,
                self.second_inference_comp,
            ],
        
            self.log_dir,
        )
