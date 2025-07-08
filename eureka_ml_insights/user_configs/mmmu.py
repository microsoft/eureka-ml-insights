"""Module providing a pipeline configuration for the MMMU dataset.

This module defines a pipeline configuration class and its method for setting up
data processing, inference, and evaluation reporting components for the MMMU dataset.
"""

import os
from typing import Any

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
from eureka_ml_insights.configs.experiment_config import ExperimentConfig
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    ASTEvalTransform,
    ColumnRename,
    CopyColumn,
    DataReader,
    HFDataReader,
    MapStringsTransform,
    MMDataLoader,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.mmmu_utils import (
    CreateMMMUPrompts,
    MMMUAll,
    MMMUTaskToCategories,
)
from eureka_ml_insights.metrics import CountAggregator, MMMUMetric


class MMMU_BASELINE_PIPELINE(ExperimentConfig):
    """Defines an ExperimentConfig pipeline for the MMMU dataset.

    There is no default model configuration for this pipeline. The model
    configuration must be provided at runtime via the command line.
    """

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        """Configures a pipeline consisting of data processing, inference, and evaluation/reporting.

        Args:
            model_config (ModelConfig):
                The model configuration to be used by the inference component.
            resume_from (str, optional):
                A path indicating where to resume the inference component from, if applicable.
            **kwargs (dict[str, Any]):
                Additional keyword arguments.

        Returns:
            PipelineConfig:
                The configured pipeline consisting of data processing, inference, and
                evaluation/reporting components.
        """

        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "MMMU/MMMU",
                    "split": "validation",
                    "tasks": MMMUAll,
                    "transform": SequenceTransform(
                        [
                            ASTEvalTransform(columns=["options"]),
                            CreateMMMUPrompts(),
                            ColumnRename(name_mapping={"answer": "ground_truth", "options": "target_options"}),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
            ignore_failure=False,
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
                            CopyColumn(column_name_src="__hf_task", column_name_dst="category"),
                            MapStringsTransform(
                                columns=["category"],
                                mapping=MMMUTaskToCategories,
                            ),
                        ]
                    ),
                },
            ),
            metric_config=MetricConfig(MMMUMetric),
            aggregator_configs=[
                AggregatorConfig(CountAggregator, {"column_names": ["MMMUMetric_result"], "normalize": True}),
                AggregatorConfig(
                    CountAggregator,
                    {"column_names": ["MMMUMetric_result"], "group_by": "category", "normalize": True},
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # Configure the pipeline
        return PipelineConfig([self.data_processing_comp, self.inference_comp, self.evalreporting_comp], self.log_dir)
