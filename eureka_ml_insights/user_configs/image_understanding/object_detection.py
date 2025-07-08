"""Module for object detection configuration classes.

This module provides example user-defined configuration classes for object detection tasks.
In order to define a new configuration, create a class that directly or indirectly inherits
from ExperimentConfig and implement the configure_pipeline method. You can also inherit
from one of the existing user-defined classes and override the necessary attributes to
reduce the amount of code you need to write.

These user-defined configuration classes are used to define the desired pipeline, which
can include any number of components. Available component options can be found in the
core module. Pass the name of the class to the main.py script to run the pipeline.
"""

import os

from eureka_ml_insights.configs import (
    AggregatorConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    MetricConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.configs.experiment_config import ExperimentConfig
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    DataReader,
    HFDataReader,
    HFJsonReader,
    MMDataLoader,
)
from eureka_ml_insights.metrics import (
    CocoDetectionAggregator,
    CocoObjectDetectionMetric,
)

from .common import LOCAL_DATA_PIPELINE


class OBJECT_DETECTION_PAIRS_PIPELINE(ExperimentConfig):
    """ExperimentConfig pipeline for the object detection (pairs condition).

    This class defines a pipeline for object detection on pairs of objects.
    There is no default model configuration; the model configuration must be
    passed in via the command line.

    Attributes:
        data_processing_comp (PromptProcessingConfig): Configuration for data processing.
        inference_comp (InferenceConfig): Configuration for inference.
        evalreporting_comp (EvalReportingConfig): Configuration for evaluation and reporting.
    """

    def configure_pipeline(self, model_config, resume_from=None):
        """Configure and return the pipeline.

        Args:
            model_config: The model configuration to be used.
            resume_from (str, optional): Path to resume training from. Defaults to None.

        Returns:
            PipelineConfig: The configured pipeline, consisting of data processing,
            inference, and evaluation/reporting components.
        """
        # Configure the data processing component.
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "microsoft/IMAGE_UNDERSTANDING",
                    "split": "val",
                    "tasks": "object_detection_pairs",
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

        target_coco_json_reader = HFJsonReader(
            repo_id="microsoft/IMAGE_UNDERSTANDING",
            repo_type="dataset",
            filename="object_detection_pairs/coco_instances.json",
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
    """ExperimentConfig pipeline for single object detection condition.

    This class extends OBJECT_DETECTION_PAIRS_PIPELINE to use the single object condition
    instead of pairs.
    """

    def configure_pipeline(self, model_config, resume_from=None):
        """Configure and return the pipeline using the single object condition.

        Args:
            model_config: The model configuration to be used.
            resume_from (str, optional): Path to resume training from. Defaults to None.

        Returns:
            PipelineConfig: The configured pipeline with the single object condition.
        """
        config = super().configure_pipeline(model_config, resume_from)
        self.data_processing_comp.data_reader_config.init_args["tasks"] = "object_detection_single"

        target_coco_json_reader = HFJsonReader(
            repo_id="microsoft/IMAGE_UNDERSTANDING",
            repo_type="dataset",
            filename="object_detection_single/coco_instances.json",
        )

        self.evalreporting_comp.metric_config.init_args["target_coco_json_reader"] = target_coco_json_reader
        self.evalreporting_comp.aggregator_configs[0].init_args["target_coco_json_reader"] = target_coco_json_reader

        return config


class OBJECT_DETECTION_PAIRS_LOCAL_PIPELINE(LOCAL_DATA_PIPELINE, OBJECT_DETECTION_PAIRS_PIPELINE):
    """Local pipeline for object detection pairs condition.

    This class combines LOCAL_DATA_PIPELINE and OBJECT_DETECTION_PAIRS_PIPELINE,
    allowing for local data paths.
    """

    def configure_pipeline(self, model_config, resume_from=None):
        """Configure and return the local pipeline for object detection pairs.

        Args:
            model_config: The model configuration to be used.
            resume_from (str, optional): Path to resume training from. Defaults to None.

        Returns:
            PipelineConfig: The configured local pipeline.
        """
        local_path = "/home/neel/data/spatial_understanding"
        return super().configure_pipeline(model_config, resume_from, local_path)


class OBJECT_DETECTION_SINGLE_LOCAL_PIPELINE(LOCAL_DATA_PIPELINE, OBJECT_DETECTION_SINGLE_PIPELINE):
    """Local pipeline for object detection single condition.

    This class combines LOCAL_DATA_PIPELINE and OBJECT_DETECTION_SINGLE_PIPELINE,
    allowing for local data paths.
    """

    def configure_pipeline(self, model_config, resume_from=None):
        """Configure and return the local pipeline for single object detection.

        Args:
            model_config: The model configuration to be used.
            resume_from (str, optional): Path to resume training from. Defaults to None.

        Returns:
            PipelineConfig: The configured local pipeline for single object detection.
        """
        local_path = "/home/neel/data/spatial_understanding"
        return super().configure_pipeline(model_config, resume_from, local_path)
