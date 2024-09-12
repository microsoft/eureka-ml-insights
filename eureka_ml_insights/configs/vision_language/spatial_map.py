import os

from eureka_ml_insights.configs.experiment_config import ExperimentConfig
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    AzureDataReader,
    AzureMMDataLoader,
    ColumnRename,
    DataLoader,
    DataReader,
    ExtractAnswerSpatialMap,
    PrependStringTransform,
    SequenceTransform,
)
from eureka_ml_insights.metrics import CaseInsensitiveMatch, CountAggregator

from ..config import (
    AggregatorConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)

"""This file contains example user defined configuration classes for the spatial map task.
In order to define a new configuration, a new class must be created that directly or indirectly
 inherits from UserDefinedConfig and the user_init method should be implemented.
You can inherit from one of the existing user defined classes below and override the necessary
attributes to reduce the amount of code you need to write.

The user defined configuration classes are used to define your desired *pipeline* that can include
any number of *component*s. Find *component* options in the core module.

Pass the name of the class to the main.py script to run the pipeline.
"""


class SPATIAL_MAP_PIPELINE(ExperimentConfig):
    """This method is used to define an eval pipeline with inference and metric report components,
    on the spatial map dataset."""

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None) -> PipelineConfig:
        # Configure the data processing component.
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                AzureDataReader,
                {
                    "account_url": "https://aifeval.blob.core.windows.net/",
                    "blob_container": "datasets",
                    "blob_name": "spatial_reason_vlm_datasets/spatial_loc_dataset/n500/questions/test/gpt4-eval-g6-n2500_QA_merged.jsonl",
                    "transform": PrependStringTransform(
                        columns="image", string="spatial_reason_vlm_datasets/spatial_loc_dataset/n500/"
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        # Configure the inference component
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                AzureMMDataLoader,
                {
                    "path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "account_url": "https://aifeval.blob.core.windows.net/",
                    "blob_container": "datasets",
                    "image_column_names": ["image"],
                },
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
        )

        # Configure the evaluation and reporting component.
        # NOTE: This component uses model-specific answer extraction that is customized for GPT-4o, Claude, and Gemini models
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
                            ExtractAnswerSpatialMap(
                                answer_column_name="model_output_raw",
                                extracted_answer_column_name="model_output",
                                question_type_column_name="question_type",
                                model_name=model_config.init_args['model_name'], # passing the model name for model-specific answer extraction
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


class SPATIAL_MAP_TEXTONLY_PIPELINE(SPATIAL_MAP_PIPELINE):
    """This class extends SPATIAL_MAP_PIPELINE to use text only data."""

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None) -> PipelineConfig:
        config = super().configure_pipeline(model_config, resume_from)
        self.data_processing_comp.data_reader_config.init_args["blob_name"] = (
            "spatial_reason_vlm_datasets/spatial_loc_dataset/n500/questions/test/gpt4-eval-g6-n2500_QA_text_only_merged.jsonl"
        )
        self.data_processing_comp.data_reader_config.init_args["transform"] = None

        self.inference_comp.data_loader_config = DataSetConfig(
            DataLoader,
            {
                "path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl"),
            },
        )

        return config


class SPATIAL_MAP_REPORTING_PIPELINE(SPATIAL_MAP_PIPELINE):
    """This method is used to define an eval pipeline with only a metric report component,
    on the spatial map dataset."""

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None) -> PipelineConfig:
        super().configure_pipeline(model_config, resume_from)
        self.evalreporting_comp.data_reader_config.init_args["path"] = resume_from
        return PipelineConfig([self.evalreporting_comp], self.log_dir)
