"""User-defined configuration classes for the geometric reasoning task on the Geometer dataset.

This module defines a specialized experiment configuration class, which configures
the pipeline for data processing, inference, and evaluation on the Geometer dataset.
"""

import os
from typing import Any

from eureka_ml_insights.configs import (
    AggregatorConfig,
    DataSetConfig,
    EvalReportingConfig,
    ExperimentConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    ColumnRename,
    DataReader,
    HFDataReader,
    MMDataLoader,
    SequenceTransform,
)
from eureka_ml_insights.metrics import CountAggregator, GeoMCQMetric


class GEOMETER_PIPELINE(ExperimentConfig):
    """Configuration class for the GEOMETER pipeline.

    This class provides methods to configure data processing, inference,
    and evaluation components for the Geometer dataset.
    """

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        """Configures the pipeline with data processing, inference, and evaluation steps.

        Args:
            model_config (ModelConfig): The model configuration to be used.
            resume_from (str, optional): A path to resume inference from a previous checkpoint. Defaults to None.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            PipelineConfig: The configured pipeline object.
        """
        # Configure the data processing component.
        data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "GeoMeterData/GeoMeter",
                    "split": "train",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={"query_text": "prompt", "target_text": "ground_truth"}),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        # Configure the inference component
        inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {"path": os.path.join(data_processing_comp.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
        )

        # Configure the evaluation and reporting component.
        evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(GeoMCQMetric),
            aggregator_configs=[
                AggregatorConfig(CountAggregator, {"column_names": ["GeoMCQMetric_result"], "normalize": True}),
                AggregatorConfig(
                    CountAggregator,
                    {"column_names": ["GeoMCQMetric_result"], "group_by": "category"},
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # Configure the pipeline
        return PipelineConfig(
            [
                data_processing_comp,
                inference_comp,
                evalreporting_comp,
            ],
            self.log_dir,
        )
