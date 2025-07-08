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
    MMDataLoader,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.spatial_utils import (
    LowerCaseNoPunctuationConvertNumbers,
)
from eureka_ml_insights.metrics import CountAggregator, ObjectRecognitionMetric

from .common import LOCAL_DATA_PIPELINE

"""Example user-defined configuration classes for the visual prompting task.

To define a new configuration, create a class that directly or indirectly
inherits from ExperimentConfig and implement the configure_pipeline method.
You can inherit from one of the existing user-defined classes below and override
the necessary attributes to reduce the amount of code required.

These user-defined configuration classes define the desired pipeline, which can
include any number of components. Find these components in the core module.

Pass the name of the class to the main.py script to run the pipeline.
"""


class VISUAL_PROMPTING_PAIRS_PIPELINE(ExperimentConfig):
    """Defines an ExperimentConfig pipeline for the visual prompting dataset (pairs condition).

    There is no model_config by default, so the model config must be passed in via the command line.

    Attributes:
        data_processing_comp (PromptProcessingConfig): Data processing component.
        inference_comp (InferenceConfig): Inference component.
        evalreporting_comp (EvalReportingConfig): Evaluation and reporting component.
    """

    def configure_pipeline(self, model_config, resume_from=None):
        """Configures the pipeline components for the visual prompting pairs condition.

        Args:
            model_config (Any): The model configuration to use.
            resume_from (Optional[str]): If specified, a path from which to resume. Defaults to None.

        Returns:
            PipelineConfig: The configured pipeline including data processing,
            inference, and evaluation components.
        """
        # Configure the data processing component.
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "microsoft/IMAGE_UNDERSTANDING",
                    "split": "val",
                    "tasks": "visual_prompting_pairs",
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


class VISUAL_PROMPTING_SINGLE_PIPELINE(VISUAL_PROMPTING_PAIRS_PIPELINE):
    """Extends VISUAL_PROMPTING_PAIRS_PIPELINE to use the single object condition."""

    def configure_pipeline(self, model_config, resume_from=None):
        """Configures the pipeline to use the single object condition.

        Args:
            model_config (Any): The model configuration to use.
            resume_from (Optional[str]): If specified, a path from which to resume. Defaults to None.

        Returns:
            PipelineConfig: The configured pipeline including data processing,
            inference, and evaluation components.
        """
        config = super().configure_pipeline(model_config, resume_from)
        self.data_processing_comp.data_reader_config.init_args["tasks"] = "visual_prompting_single"
        return config


class VISUAL_PROMPTING_PAIRS_LOCAL_PIPELINE(LOCAL_DATA_PIPELINE, VISUAL_PROMPTING_PAIRS_PIPELINE):
    """Pipeline for visual prompting pairs in a local environment.

    Uses a local path for data and inherits from LOCAL_DATA_PIPELINE and
    VISUAL_PROMPTING_PAIRS_PIPELINE.
    """

    def configure_pipeline(self, model_config, resume_from=None):
        """Configures the pipeline for local visual prompting pairs data.

        Args:
            model_config (Any): The model configuration to use.
            resume_from (Optional[str]): If specified, a path from which to resume. Defaults to None.
            local_path (str): Local data path (injected by super call).

        Returns:
            PipelineConfig: The configured pipeline including data processing,
            inference, and evaluation components.
        """
        local_path = "/home/neel/data/spatial_understanding"
        return super().configure_pipeline(model_config, resume_from, local_path)


class VISUAL_PROMPTING_SINGLE_LOCAL_PIPELINE(LOCAL_DATA_PIPELINE, VISUAL_PROMPTING_SINGLE_PIPELINE):
    """Pipeline for visual prompting single object condition in a local environment.

    Uses a local path for data and inherits from LOCAL_DATA_PIPELINE and
    VISUAL_PROMPTING_SINGLE_PIPELINE.
    """

    def configure_pipeline(self, model_config, resume_from=None):
        """Configures the pipeline for the local single object condition.

        Args:
            model_config (Any): The model configuration to use.
            resume_from (Optional[str]): If specified, a path from which to resume. Defaults to None.
            local_path (str): Local data path (injected by super call).

        Returns:
            PipelineConfig: The configured pipeline including data processing,
            inference, and evaluation components.
        """
        local_path = "/home/neel/data/spatial_understanding"
        return super().configure_pipeline(model_config, resume_from, local_path)
