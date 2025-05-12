import os
from typing import Any

from eureka_ml_insights.configs import (
    DataSetConfig,
    ExperimentConfig,
    InferenceConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.core import Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    ColumnRename,
    HFDataReader,
    MMDataLoader,
    MultiplyTransform,
    SequenceTransform,
)

"""This file contains user defined configuration classes for the SAT benchmark.
"""


class NPHARD_SAT_PIPELINE(ExperimentConfig):
    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        # Configure the data processing component.
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "GeoMeterData/nphard_sat2",
                    "split": "train",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={"query_text": "prompt", "target_text": "ground_truth"}),
                        ]
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/nphard_sat_templates/Template_sat_cot.jinja"
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        # Configure the inference component
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {"path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
            max_concurrent=5,
        )

        # Configure the pipeline
        return PipelineConfig(
            [self.data_processing_comp, self.inference_comp],
            self.log_dir,
        )


class NPHARD_SAT_PIPELINE_MULTIPLE_RUNS(NPHARD_SAT_PIPELINE):
    """This class specifies the config for running SAT benchmark n repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=1)
        )
        return pipeline
