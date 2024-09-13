import os

from eureka_ml_insights.configs.experiment_config import ExperimentConfig
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    AzureDataReader,
    AzureJsonReader,
    AzureMMDataLoader,
    ColumnRename,
    CopyColumn,
    DataReader,
    PrependStringTransform,
    SequenceTransform,
)
from eureka_ml_insights.metrics import (
    CocoDetectionAggregator,
    CocoObjectDetectionMetric,
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

"""This file contains example user defined configuration classes for the object detection task.
In order to define a new configuration, a new class must be created that directly or indirectly
 inherits from ExperimentConfig and the configure_pipeline method should be implemented.
You can inherit from one of the existing user defined classes below and override the necessary
attributes to reduce the amount of code you need to write.

The user defined configuration classes are used to define your desired *pipeline* that can include
any number of *component*s. Find *component* options in the core module.

Pass the name of the class to the main.py script to run the pipeline.
"""


class OBJECT_DETECTION_PAIRS_PIPELINE(ExperimentConfig):
    """
    This defines an ExperimentConfig pipeline for the object detection dataset, pairs condition.
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
                    "blob_name": "msr_aif_object_detection_pairs/object_detection_val_long_prompt.jsonl",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={"query_text": "prompt", "target_text": "ground_truth"}),
                            CopyColumn(column_name_src="images", column_name_dst="images_prepended"),
                            PrependStringTransform(
                                columns="images_prepended", string="msr_aif_object_detection_pairs/"
                            ),
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
                    "image_column_names": ["images_prepended"],
                },
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
        )

        target_coco_json_reader = AzureJsonReader(
            account_url="https://aifeval.blob.core.windows.net/",
            blob_container="datasets",
            blob_name="msr_aif_object_detection_pairs/coco_instances.json",
        )

        # Configure the evaluation and reporting component.
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(
                CocoObjectDetectionMetric,
                {"target_coco_json_reader": target_coco_json_reader},
            ),
            aggregator_configs=[
                AggregatorConfig(
                    CocoDetectionAggregator,
                    {
                        "column_names": ["CocoObjectDetectionMetric_result"],
                        "target_coco_json_reader": target_coco_json_reader,
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # Configure the pipeline
        return PipelineConfig([self.data_processing_comp, self.inference_comp, self.evalreporting_comp], self.log_dir)


class OBJECT_DETECTION_SINGLE_PIPELINE(OBJECT_DETECTION_PAIRS_PIPELINE):
    """This class extends OBJECT_DETECTION_PAIRS_PIPELINE to use the single object condition."""

    def configure_pipeline(self, model_config, resume_from=None):
        config = super().configure_pipeline(model_config, resume_from)
        self.data_processing_comp.data_reader_config.init_args["blob_name"] = (
            "msr_aif_object_detection_single/object_detection_val_long_prompt.jsonl"
        )
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms[
            2
        ].string = "msr_aif_object_detection_single/"

        target_coco_json_reader = AzureJsonReader(
            account_url="https://aifeval.blob.core.windows.net/",
            blob_container="datasets",
            blob_name="msr_aif_object_detection_single/coco_instances.json",
        )

        self.evalreporting_comp.metric_config.init_args["target_coco_json_reader"] = target_coco_json_reader
        self.evalreporting_comp.aggregator_configs[0].init_args["target_coco_json_reader"] = target_coco_json_reader

        return config


class OBJECT_DETECTION_PAIRS_LOCAL_PIPELINE(LOCAL_DATA_PIPELINE, OBJECT_DETECTION_PAIRS_PIPELINE):
    def configure_pipeline(self, model_config, resume_from=None):
        local_path = "/home/neel/data/spatial_understanding"
        return super().configure_pipeline(model_config, resume_from, local_path)


class OBJECT_DETECTION_SINGLE_LOCAL_PIPELINE(LOCAL_DATA_PIPELINE, OBJECT_DETECTION_SINGLE_PIPELINE):
    def configure_pipeline(self, model_config, resume_from=None):
        local_path = "/home/neel/data/spatial_understanding"
        return super().configure_pipeline(model_config, resume_from, local_path)
