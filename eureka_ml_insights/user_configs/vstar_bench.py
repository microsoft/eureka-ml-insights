import os
from typing import Any

from eureka_ml_insights.configs.experiment_config import ExperimentConfig
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing, DataProcessing

from eureka_ml_insights.data_utils import (
    HFDataReader,  
    MMDataLoader,
    DataLoader,
    DataReader,
    SequenceTransform,
    ColumnRename,
    AddColumn,
)
from eureka_ml_insights.metrics import CountAggregator, SubstringExistsMatch

from eureka_ml_insights.configs import (
    AggregatorConfig,
    DataProcessingConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.configs.model_configs import OAI_GPT4_1106_PREVIEW_CONFIG as PERSONAL_GPT4O

"""This file contains example user defined configuration classes for the V*Bench task.
In order to define a new configuration, a new class must be created that directly or indirectly
 inherits from UserDefinedConfig and the user_init method should be implemented.
You can inherit from one of the existing user defined classes below and override the necessary
attributes to reduce the amount of code you need to write.

The user defined configuration classes are used to define your desired *pipeline* that can include
any number of *component*s. Find *component* options in the core module.

Pass the name of the class to the main.py script to run the pipeline.
"""


class VSTAR_BENCH_PIPELINE(ExperimentConfig):
    """This method is used to define an eval pipeline with inference and metric report components,
    on the V*Bench dataset."""

    def configure_pipeline(
        self, model_config: ModelConfig,
        resume_from: str = None,
        **kwargs: dict[str, Any],
    ) -> PipelineConfig:

        # Get the user provided LLM judge configuration, defaulting to PERSONAL_GPT4O if not provided.
        LLM_JUDGE_CONFIG = kwargs.get("llm_judge_config", PERSONAL_GPT4O)

        # Download V*Bench from HuggingFace
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "tmlabonte/vstar_bench",
                    "split": "test",
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        # Perform inference with desired model on V*Bench.
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {
                    "path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl"),
                },
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
            max_concurrent=8,
        )

        # Prepare inference result for LLM answer extraction
        self.preeval_data_post_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={
                                "prompt": "initial_prompt",
                                "model_output": "model_output_raw",
                            }),
                            AddColumn(column_name="prompt"),
                        ]
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__),
                "../prompt_templates/vstar_bench_templates/extract_answer.jinja",
            ),
            output_dir=os.path.join(self.log_dir, "preeval_data_post_processing_output"),
        )

        # Extract answer using LLM
        self.llm_answer_extract_comp = InferenceConfig(
            component_type=Inference,
            model_config=LLM_JUDGE_CONFIG,
            data_loader_config=DataSetConfig(
                DataLoader,
                {"path": os.path.join(self.preeval_data_post_processing_comp.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(self.log_dir, "llm_answer_extract_inference_result"),
            max_concurrent=8,
        )

        # Evaluate extracted answer
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.llm_answer_extract_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(SubstringExistsMatch),
            aggregator_configs=[
                AggregatorConfig(
                    CountAggregator, {"column_names": ["SubstringExistsMatch_result"], "normalize": True}
                ),
                AggregatorConfig(
                    CountAggregator,
                    {"column_names": ["SubstringExistsMatch_result"], "group_by": "category", "normalize": True},
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # Configure the pipeline
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.preeval_data_post_processing_comp,
                self.llm_answer_extract_comp,
                self.evalreporting_comp,
            ],
            self.log_dir,
        )
