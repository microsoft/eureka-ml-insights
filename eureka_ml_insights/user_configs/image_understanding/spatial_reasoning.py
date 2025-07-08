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
    AddColumnAndData,
    ASTEvalTransform,
    DataReader,
    HFDataReader,
    MMDataLoader,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.spatial_utils import (
    LowerCaseNoPunctuationConvertNumbers,
)
from eureka_ml_insights.metrics import (
    CountAggregator,
    SpatialAndLayoutReasoningMetric,
)

from .common import LOCAL_DATA_PIPELINE

"""Example user-defined configuration classes for the spatial reasoning task.

To define a new configuration, create a class that directly or indirectly inherits
from ExperimentConfig and implement the configure_pipeline method. You can inherit
from one of the existing user-defined classes below and override the necessary
attributes to reduce the amount of code you need to write.

These user-defined configuration classes define a pipeline that can include any
number of components. See the core module for possible components.

To run the pipeline, pass the name of the class to the main.py script.
"""


class SPATIAL_REASONING_PAIRS_PIPELINE(ExperimentConfig):
    """Defines an ExperimentConfig pipeline for the spatial reasoning dataset (pairs condition).

    This class creates a pipeline without a default model_config. The model_config
    must be passed in via the command line.
    """

    def configure_pipeline(self, model_config, resume_from=None):
        """Configures the pipeline components for data processing, inference, and evaluation.

        Args:
            model_config: The model configuration to be used by the Inference component.
            resume_from: Checkpoint or step to resume from, if any. Defaults to None.

        Returns:
            PipelineConfig: The configuration containing the sequence of pipeline components.
        """
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "microsoft/IMAGE_UNDERSTANDING",
                    "split": "val",
                    "tasks": "spatial_reasoning_lrtb_pairs",
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

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

        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            AddColumnAndData("target_options", "['left', 'right', 'above', 'below']"),
                            ASTEvalTransform(columns=["target_options"]),
                            LowerCaseNoPunctuationConvertNumbers(
                                columns=["ground_truth", "model_output", "target_options"]
                            ),
                        ]
                    ),
                },
            ),
            metric_config=MetricConfig(SpatialAndLayoutReasoningMetric),
            aggregator_configs=[
                AggregatorConfig(
                    CountAggregator, {"column_names": ["SpatialAndLayoutReasoningMetric_result"], "normalize": True}
                ),
                AggregatorConfig(
                    CountAggregator,
                    {"column_names": ["SpatialAndLayoutReasoningMetric_result"], "group_by": "ground_truth"},
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        return PipelineConfig([self.data_processing_comp, self.inference_comp, self.evalreporting_comp], self.log_dir)


class SPATIAL_REASONING_SINGLE_PIPELINE(SPATIAL_REASONING_PAIRS_PIPELINE):
    """Extends SPATIAL_REASONING_PAIRS_PIPELINE to use the single object condition."""

    def configure_pipeline(self, model_config, resume_from=None):
        """Configures the pipeline for the single object condition.

        Args:
            model_config: The model configuration to be used by the Inference component.
            resume_from: Checkpoint or step to resume from, if any. Defaults to None.

        Returns:
            PipelineConfig: The updated pipeline configuration for the single object condition.
        """
        config = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        self.data_processing_comp.data_reader_config.init_args["tasks"] = "spatial_reasoning_lrtb_single"
        self.evalreporting_comp.data_reader_config.init_args["transform"].transforms[
            0
        ].data = "['left', 'right', 'top', 'bottom']"
        return config


class SPATIAL_REASONING_PAIRS_LOCAL_PIPELINE(LOCAL_DATA_PIPELINE, SPATIAL_REASONING_PAIRS_PIPELINE):
    """Defines a local data variant of the SPATIAL_REASONING_PAIRS_PIPELINE."""

    def configure_pipeline(self, model_config, resume_from=None):
        """Configures the pipeline for local data in the pairs condition.

        Args:
            model_config: The model configuration to be used by the Inference component.
            resume_from: Checkpoint or step to resume from, if any. Defaults to None.

        Returns:
            PipelineConfig: The pipeline configuration using local data for pairs condition.
        """
        local_path = "/home/neel/data/spatial_understanding"
        return super().configure_pipeline(model_config, resume_from, local_path)


class SPATIAL_REASONING_SINGLE_LOCAL_PIPELINE(LOCAL_DATA_PIPELINE, SPATIAL_REASONING_SINGLE_PIPELINE):
    """Defines a local data variant of the SPATIAL_REASONING_SINGLE_PIPELINE."""

    def configure_pipeline(self, model_config, resume_from=None):
        """Configures the pipeline for local data in the single object condition.

        Args:
            model_config: The model configuration to be used by the Inference component.
            resume_from: Checkpoint or step to resume from, if any. Defaults to None.

        Returns:
            PipelineConfig: The pipeline configuration using local data for single condition.
        """
        local_path = "/home/neel/data/spatial_understanding"
        return super().configure_pipeline(model_config, resume_from, local_path)
