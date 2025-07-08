"""
Module for configuring and running IFEval benchmarks.

This module defines several pipeline configurations for benchmarking models
using the IFEval benchmark. It includes classes for setting up data processing,
inference, evaluation, and reporting components.
"""

import os
from typing import Any

from eureka_ml_insights.configs import (
    AggregatorConfig,
    DataProcessingConfig,
    DataSetConfig,
    EvalReportingConfig,
    ExperimentConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.core import DataProcessing, Inference, PromptProcessing
from eureka_ml_insights.core.eval_reporting import EvalReporting
from eureka_ml_insights.data_utils.data import (
    DataLoader,
    DataReader,
    HFDataReader,
)
from eureka_ml_insights.data_utils.transform import (
    CopyColumn,
    MultiplyTransform,
    RunPythonTransform,
    SequenceTransform,
)
from eureka_ml_insights.metrics.ifeval_metrics import IFEvalMetric
from eureka_ml_insights.metrics.reports import (
    AverageAggregator,
    BiLevelAggregator,
    TwoColumnSumAverageAggregator,
)


class IFEval_PIPELINE(ExperimentConfig):
    """
    Configures and runs the IFEval benchmark on any model.

    This class extends ExperimentConfig to define a pipeline consisting of
    data processing, inference, evaluation, post-processing, and instruction-level
    evaluation components.
    """

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        """
        Configures the pipeline for the IFEval benchmark.

        Args:
            model_config (ModelConfig): The configuration for the model to be used.
            resume_from (str, optional): Path to a checkpoint to resume from. Defaults to None.
            **kwargs (dict[str, Any]): Additional keyword arguments for pipeline configuration.

        Returns:
            PipelineConfig: The configured pipeline.
        """

        # data preprocessing
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "google/IFEval",
                    "split": "train",
                    "transform": SequenceTransform([MultiplyTransform(n_repeats=1)]),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        # inference component
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                DataLoader,
                {"path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
            max_concurrent=20,
        )

        # Configure the evaluation and reporting component for evaluation and dataset level aggregation
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [CopyColumn(column_name_src="model_output", column_name_dst="response")]
                    ),
                },
            ),
            metric_config=MetricConfig(IFEvalMetric),
            aggregator_configs=[
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": [
                            "IFEvalMetric_strict_follow_all_instructions",
                            "IFEvalMetric_loose_follow_all_instructions",
                        ],
                        "filename_base": "IFEvalAccuracyMetrics_SeparateRuns",
                        "group_by": "data_repeat_id",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "IFEvalMetric_strict_follow_all_instructions",
                            "IFEvalMetric_loose_follow_all_instructions",
                        ],
                        "filename_base": "IFEvalAccuracyMetrics_Aggregated",
                        "first_groupby": "data_repeat_id",
                        "agg_fn": "mean",
                    },
                ),
                AggregatorConfig(
                    TwoColumnSumAverageAggregator,
                    {
                        "numerator_column_name": "IFEvalMetric_strict_follow_instruction_list_sum",
                        "denominator_column_name": "IFEvalMetric_strict_instruction_list_len",
                        "filename_base": "IFEvalStrictInfoFollowRateMetric_SeparateRuns",
                        "group_by": "data_repeat_id",
                    },
                ),
                AggregatorConfig(
                    TwoColumnSumAverageAggregator,
                    {
                        "numerator_column_name": "IFEvalMetric_loose_follow_instruction_list_sum",
                        "denominator_column_name": "IFEvalMetric_loose_instruction_list_len",
                        "filename_base": "IFEvalLooseInfoFollowRateMetric_SeparateRuns",
                        "group_by": "data_repeat_id",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # Configure the eval post processing component to explode instruction types
        self.eval_post_processing_comp = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.evalreporting_comp.output_dir, "metric_results.jsonl"),
                    "format": ".jsonl",
                    "transform": RunPythonTransform(
                        "df = df.explode(['instruction_id_list', 'IFEvalMetric_tier0_instructions', "
                        " 'IFEvalMetric_strict_follow_instruction_list', "
                        " 'IFEvalMetric_loose_follow_instruction_list']) "
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "metric_post_processing_output"),
        )

        # Configure the reporting component for instruction level aggregation
        self.instruction_level_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.eval_post_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            aggregator_configs=[
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "IFEvalMetric_strict_follow_instruction_list",
                            "IFEvalMetric_loose_follow_instruction_list",
                        ],
                        "first_groupby": ["data_repeat_id", "instruction_id_list"],
                        "second_groupby": "instruction_id_list",
                        "agg_fn": "mean",
                        "filename_base": "IFEvalAccuracyMetrics_GroupByInstructionID",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "IFEvalMetric_strict_follow_instruction_list",
                            "IFEvalMetric_loose_follow_instruction_list",
                        ],
                        "first_groupby": ["data_repeat_id", "IFEvalMetric_tier0_instructions"],
                        "second_groupby": "IFEvalMetric_tier0_instructions",
                        "agg_fn": "mean",
                        "filename_base": "IFEvalAccuracyMetrics_GroupByTier0Instructions",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "instruction_level_eval_report"),
        )

        # Configure the pipeline
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.evalreporting_comp,
                self.eval_post_processing_comp,
                self.instruction_level_evalreporting_comp,
            ],
            self.log_dir,
        )


class IFEval_Parallel_PIPELINE(IFEval_PIPELINE):
    """
    Configures and runs the IFEval benchmark multiple times in parallel.

    This class extends IFEval_PIPELINE to repeat the data multiple times during
    data processing, enabling parallel evaluations.
    """

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        """
        Configures the pipeline for the parallel IFEval benchmark.

        Args:
            model_config (ModelConfig): The configuration for the model to be used.
            resume_from (str, optional): Path to a checkpoint to resume from. Defaults to None.
            **kwargs (dict[str, Any]): Additional keyword arguments for pipeline configuration.

        Returns:
            PipelineConfig: The configured pipeline with repeated data transforms.
        """

        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms[-1] = MultiplyTransform(
            n_repeats=3
        )
        return pipeline


class IFEval_Phi_Parallel_PIPELINE(IFEval_Parallel_PIPELINE):
    """
    Configures and runs the IFEval benchmark for Phi-reasoning models.

    This class extends IFEval_Parallel_PIPELINE and adds custom post-processing
    to handle 'thinking tokens' in model responses.
    """

    def configure_pipeline(
        self,
        model_config: ModelConfig,
        resume_from: str = None,
        thinking_token: str = "</think>",
        **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        """
        Configures the pipeline for the Phi-reasoning IFEval benchmark.

        Args:
            model_config (ModelConfig): The configuration for the model to be used.
            resume_from (str, optional): Path to a checkpoint to resume from. Defaults to None.
            thinking_token (str, optional): The token that precedes hidden reasoning logic. Defaults to "</think>".
            **kwargs (dict[str, Any]): Additional keyword arguments for pipeline configuration.

        Returns:
            PipelineConfig: The configured pipeline with custom response processing.
        """

        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # eval data processing
        self.evalreporting_comp.data_reader_config.init_args["transform"].transforms.append(
            RunPythonTransform(
                "df['response'] = df['response'].apply(lambda x: x.split('{token}')[-1] if '{token}' in x else x)".format(
                    token=thinking_token
                )
            )
        )
        return pipeline
