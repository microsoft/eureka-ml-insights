import os

from eureka_ml_insights.configs.experiment_config import ExperimentConfig
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    HFDataReader,    
    MMDataLoader,
    ColumnRename,
    DataLoader,
    DataReader,
    ExtractAnswerGrid,
    PrependStringTransform,
    SequenceTransform,
)
from eureka_ml_insights.metrics import CaseInsensitiveMatch, CountAggregator
from eureka_ml_insights.configs import (
    AggregatorConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)

"""This file contains example user defined configuration classes for the grid counting task.
In order to define a new configuration, a new class must be created that directly or indirectly
 inherits from UserDefinedConfig and the user_init method should be implemented.
You can inherit from one of the existing user defined classes below and override the necessary
attributes to reduce the amount of code you need to write.

The user defined configuration classes are used to define your desired *pipeline* that can include
any number of *component*s. Find *component* options in the core module.

Pass the name of the class to the main.py script to run the pipeline.
"""


class SPATIAL_GRID_PIPELINE(ExperimentConfig):
    """This method is used to define an eval pipeline with inference and metric report components,
    on the grid counting dataset."""

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None) -> PipelineConfig:
        # Configure the data processing component.
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "microsoft/VISION_LANGUAGE",
                    "split": "val",
                    "tasks": "spatial_grid",
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        # Configure the inference component
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
        )

        # Configure the evaluation and reporting component.
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={"model_output": "model_output_raw"}),
                            ExtractAnswerGrid(
                                answer_column_name="model_output_raw",
                                extracted_answer_column_name="model_output",
                                question_type_column_name="question_type",
                                mode="animal",
                            ),
                        ],
                    ),
                },
            ),
            metric_config=MetricConfig(CaseInsensitiveMatch),
            aggregator_configs=[
                AggregatorConfig(CountAggregator, {"column_names": ["CaseInsensitiveMatch_result"], "normalize": True}),
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": ["CaseInsensitiveMatch_result"],
                        "group_by": "task",
                        "normalize": True,
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # Configure the pipeline
        return PipelineConfig([self.data_processing_comp, self.inference_comp, self.evalreporting_comp], self.log_dir)


class SPATIAL_GRID_TEXTONLY_PIPELINE(SPATIAL_GRID_PIPELINE):
    """This class extends SPATIAL_GRID_PIPELINE to use text only data."""

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None) -> PipelineConfig:
        config = super().configure_pipeline(model_config, resume_from)
        self.data_processing_comp.data_reader_config.init_args["tasks"] = (
            "spatial_grid_text_only"
        )
        return config


class SPATIAL_GRID_REPORTING_PIPELINE(SPATIAL_GRID_PIPELINE):
    """This method is used to define an eval pipeline with only a metric report component,
    on the grid counting dataset."""

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None) -> PipelineConfig:
        super().configure_pipeline(model_config, resume_from)
        self.evalreporting_comp.data_reader_config.init_args["path"] = resume_from
        return PipelineConfig([self.evalreporting_comp], self.log_dir)
