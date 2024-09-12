import os

from eureka_ml_insights.configs.experiment_config import ExperimentConfig
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    AzureDataReader,
    AzureMMDataLoader,
    ColumnRename,
    DataReader,
    PrependStringTransform,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.spatial_utils import (
    LowerCaseNoPunctuationConvertNumbers,
)
from eureka_ml_insights.metrics import (
    CountAggregator,
    ObjectRecognitionMetric,
)

from ..config import (
    AggregatorConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    MetricConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from .common import LOCAL_DATA_PIPELINE

"""This file contains example user defined configuration classes for the object recognition task.
In order to define a new configuration, a new class must be created that directly or indirectly
 inherits from ExperimentConfig and the configure_pipeline method should be implemented.
You can inherit from one of the existing user defined classes below and override the necessary
attributes to reduce the amount of code you need to write.

The user defined configuration classes are used to define your desired *pipeline* that can include
any number of *component*s. Find *component* options in the core module.

Pass the name of the class to the main.py script to run the pipeline.
"""


class OBJECT_RECOGNITION_PAIRS_PIPELINE(ExperimentConfig):
    """
    This defines an ExperimentConfig pipeline for the object recognition dataset, pairs condition.
    There is no model_config by default and the model config must be passed in via command lime.
    """

    def configure_pipeline(self, model_config, resume_from=None):
        # Configure the data processing component.
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                AzureDataReader,
                {
                    "account_url": "https://aifeval.blob.core.windows.net/",
                    "blob_container": "datasets",
                    "blob_name": "msr_aif_spatial_reasoning_lrtb_pairs/recognition_val.jsonl",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={"query_text": "prompt", "target_text": "ground_truth"}),
                            PrependStringTransform(columns="images", string="msr_aif_spatial_reasoning_lrtb_pairs/"),
                        ]
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
                    "image_column_names": ["images"],
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
                            LowerCaseNoPunctuationConvertNumbers(columns=["model_output"]),
                        ]
                    ),
                },
            ),
            metric_config=MetricConfig(ObjectRecognitionMetric),
            aggregator_configs=[
                AggregatorConfig(
                    CountAggregator, {"column_names": ["ObjectRecognitionMetric_result"], "normalize": True}
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # Configure the pipeline
        return PipelineConfig([self.data_processing_comp, self.inference_comp, self.evalreporting_comp], self.log_dir)


class OBJECT_RECOGNITION_SINGLE_PIPELINE(OBJECT_RECOGNITION_PAIRS_PIPELINE):
    """This class extends OBJECT_RECOGNITION_PAIRS_PIPELINE to use the single object condition."""

    def configure_pipeline(self, model_config, resume_from=None):
        config = super().configure_pipeline(model_config, resume_from)
        self.data_processing_comp.data_reader_config.init_args["blob_name"] = (
            "msr_aif_spatial_reasoning_lrtb_single/recognition_val.jsonl"
        )
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms[
            1
        ].string = "msr_aif_spatial_reasoning_lrtb_single/"
        return config


class OBJECT_RECOGNITION_PAIRS_LOCAL_PIPELINE(LOCAL_DATA_PIPELINE, OBJECT_RECOGNITION_PAIRS_PIPELINE):
    def configure_pipeline(self, model_config, resume_from=None):
        local_path = "/home/neel/data/spatial_understanding"
        return super().configure_pipeline(model_config, resume_from, local_path)


class OBJECT_RECOGNITION_SINGLE_LOCAL_PIPELINE(LOCAL_DATA_PIPELINE, OBJECT_RECOGNITION_SINGLE_PIPELINE):
    def configure_pipeline(self, model_config, resume_from=None):
        local_path = "/home/neel/data/spatial_understanding"
        return super().configure_pipeline(model_config, resume_from, local_path)
