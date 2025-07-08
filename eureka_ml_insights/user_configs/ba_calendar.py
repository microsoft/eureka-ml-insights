"""This module defines pipeline configurations for BA Calendar tasks.

It includes multiple classes that define different pipeline configurations 
for data processing, inference, evaluation, and reporting for BA Calendar 
tasks.
"""

import os
from typing import Any

from eureka_ml_insights.core import Inference, PromptProcessing
from eureka_ml_insights.core.data_processing import DataProcessing
from eureka_ml_insights.core.eval_reporting import EvalReporting
from eureka_ml_insights.data_utils.ba_calendar_utils import (
    BA_Calendar_ExtractAnswer,
)
from eureka_ml_insights.data_utils.data import (
    DataLoader,
    DataReader,
    HFDataReader,
)
from eureka_ml_insights.data_utils.transform import (
    AddColumn,
    AddColumnAndData,
    ColumnRename,
    CopyColumn,
    ExtractUsageTransform,
    MajorityVoteTransform,
    MultiplyTransform,
    RunPythonTransform,
    SequenceTransform,
)
from eureka_ml_insights.metrics.ba_calendar_metrics import BACalendarMetric
from eureka_ml_insights.metrics.reports import (
    AverageAggregator,
    BiLevelAggregator,
    CountAggregator,
)

from ..configs.config import (
    AggregatorConfig,
    DataProcessingConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from ..configs.experiment_config import ExperimentConfig


class BA_Calendar_PIPELINE(ExperimentConfig):
    """Specifies the configuration for running any benchmark on any model.

    BA_Calendar_PIPELINE extends the ExperimentConfig class. It defines
    the data processing, inference, evaluation reporting, and other components
    needed to run a BA Calendar pipeline.
    """

    def configure_pipeline(self, model_config=None, resume_from=None, resume_logdir=None, **kwargs) -> PipelineConfig:
        """Configures the pipeline components and returns a PipelineConfig object.

        Args:
            model_config (Optional[ModelConfig]): The model configuration object.
            resume_from (Optional[str]): Directory to resume from. Defaults to None.
            resume_logdir (Optional[str]): Directory of logs to resume from. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            PipelineConfig: The configured pipeline containing data processing,
                inference, evaluation reporting, and other components.
        """
        # data preprocessing
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/ba_calendar_templates/calendar_scheduling_cot.jinja"
            ),
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "microsoft/ba-calendar",
                    "split": "test",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={"task_prompt": "prompt"}),
                            # SamplerTransform(random_seed=5, sample_count=10),
                            MultiplyTransform(n_repeats=1),
                        ]
                    ),
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
            max_concurrent=1,
        )

        if resume_logdir:
            self.log_dir = resume_from.split("/")[0 : len(resume_from.split("/")) - 1]

        # Configure the evaluation and reporting component for evaluation and dataset level aggregation
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            ExtractUsageTransform(model_config),
                            ColumnRename(
                                name_mapping={
                                    "model_output": "raw_output",
                                }
                            ),
                            AddColumn("model_output"),
                            BA_Calendar_ExtractAnswer("raw_output", "model_output"),
                        ]
                    ),
                },
            ),
            metric_config=MetricConfig(BACalendarMetric),
            aggregator_configs=[
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": [
                            "BACalendarMetric_all_correct",
                            "BACalendarMetric_fraction_passed",
                            "BACalendarMetric_availability_programmatic_check",
                            "BACalendarMetric_meeting_duration_programmatic_check",
                            "BACalendarMetric_buffer_time_programmatic_check",
                            "BACalendarMetric_no_weekends_programmatic_check",
                            "BACalendarMetric_time_restrictions_programmatic_check",
                            "BACalendarMetric_specific_times_programmatic_check",
                            "BACalendarMetric_priority_programmatic_check",
                        ],
                        "filename_base": "OverallMetrics_Separate_Runs",
                        "group_by": "data_repeat_id",
                    },
                ),
                # the next three reports take the average and std for all repeats
                # the resulting numbers are the average and std of N pass@1 scores, where N is number of repeats
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "BACalendarMetric_all_correct",
                            "BACalendarMetric_fraction_passed",
                            "BACalendarMetric_availability_programmatic_check",
                            "BACalendarMetric_meeting_duration_programmatic_check",
                            "BACalendarMetric_buffer_time_programmatic_check",
                            "BACalendarMetric_no_weekends_programmatic_check",
                            "BACalendarMetric_time_restrictions_programmatic_check",
                            "BACalendarMetric_specific_times_programmatic_check",
                            "BACalendarMetric_priority_programmatic_check",
                        ],
                        "first_groupby": "data_repeat_id",
                        "filename_base": "OverallMetrics_Avg",
                        "agg_fn": "mean",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "BACalendarMetric_all_correct",
                            "BACalendarMetric_fraction_passed",
                            "BACalendarMetric_availability_programmatic_check",
                            "BACalendarMetric_meeting_duration_programmatic_check",
                            "BACalendarMetric_buffer_time_programmatic_check",
                            "BACalendarMetric_no_weekends_programmatic_check",
                            "BACalendarMetric_time_restrictions_programmatic_check",
                            "BACalendarMetric_specific_times_programmatic_check",
                            "BACalendarMetric_priority_programmatic_check",
                        ],
                        "first_groupby": ["data_repeat_id", "BACalendarMetric_constrainedness_bucket"],
                        "second_groupby": "BACalendarMetric_constrainedness_bucket",
                        "filename_base": "OverallMetrics_Avg_by_constrainedness",
                        "agg_fn": "mean",
                    },
                ),
                # reports for average completion usage
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": "data_point_id",
                        "filename_base": "UsageCompletion_AllRuns",
                        "agg_fn": "mean",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": ["data_point_id", "BACalendarMetric_constrainedness_bucket"],
                        "second_groupby": "BACalendarMetric_constrainedness_bucket",
                        "filename_base": "UsageCompletion_by_constrainedness_AllRuns",
                        "agg_fn": "mean",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # Aggregate the results by best of n
        self.bon_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.evalreporting_comp.output_dir, "metric_results.jsonl"),
                    "format": ".jsonl",
                },
            ),
            aggregator_configs=[
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "BACalendarMetric_all_correct",
                            "BACalendarMetric_fraction_passed",
                            "BACalendarMetric_availability_programmatic_check",
                            "BACalendarMetric_meeting_duration_programmatic_check",
                            "BACalendarMetric_buffer_time_programmatic_check",
                            "BACalendarMetric_no_weekends_programmatic_check",
                            "BACalendarMetric_time_restrictions_programmatic_check",
                            "BACalendarMetric_specific_times_programmatic_check",
                            "BACalendarMetric_priority_programmatic_check",
                        ],
                        "first_groupby": "data_point_id",
                        "filename_base": "OverallMetrics_BestofN",
                        "normalize": True,
                        "agg_fn": "max",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "BACalendarMetric_all_correct",
                            "BACalendarMetric_fraction_passed",
                            "BACalendarMetric_availability_programmatic_check",
                            "BACalendarMetric_meeting_duration_programmatic_check",
                            "BACalendarMetric_buffer_time_programmatic_check",
                            "BACalendarMetric_no_weekends_programmatic_check",
                            "BACalendarMetric_time_restrictions_programmatic_check",
                            "BACalendarMetric_specific_times_programmatic_check",
                            "BACalendarMetric_priority_programmatic_check",
                        ],
                        "first_groupby": "data_point_id",
                        "second_groupby": "BACalendarMetric_constrainedness_bucket",
                        "filename_base": "OverallMetrics_BestOfN_by_constrainedness",
                        "normalize": True,
                        "agg_fn": "max",
                    },
                ),
                # aggregates results by data_point_id and takes the sum of usage for completion tokens
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": "data_point_id",
                        "filename_base": "UsageCompletion_BestOfN",
                        "agg_fn": "sum",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "bestofn_eval_report"),
        )
        # Aggregate the results by worst of n
        self.won_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.evalreporting_comp.output_dir, "metric_results.jsonl"),
                    "format": ".jsonl",
                },
            ),
            aggregator_configs=[
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "BACalendarMetric_all_correct",
                            "BACalendarMetric_fraction_passed",
                            "BACalendarMetric_availability_programmatic_check",
                            "BACalendarMetric_meeting_duration_programmatic_check",
                            "BACalendarMetric_buffer_time_programmatic_check",
                            "BACalendarMetric_no_weekends_programmatic_check",
                            "BACalendarMetric_time_restrictions_programmatic_check",
                            "BACalendarMetric_specific_times_programmatic_check",
                            "BACalendarMetric_priority_programmatic_check",
                        ],
                        "first_groupby": "data_point_id",
                        "filename_base": "OverallMetrics_WorstofN",
                        "normalize": True,
                        "agg_fn": "min",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "BACalendarMetric_all_correct",
                            "BACalendarMetric_fraction_passed",
                            "BACalendarMetric_availability_programmatic_check",
                            "BACalendarMetric_meeting_duration_programmatic_check",
                            "BACalendarMetric_buffer_time_programmatic_check",
                            "BACalendarMetric_no_weekends_programmatic_check",
                            "BACalendarMetric_time_restrictions_programmatic_check",
                            "BACalendarMetric_specific_times_programmatic_check",
                            "BACalendarMetric_priority_programmatic_check",
                        ],
                        "first_groupby": "data_point_id",
                        "second_groupby": "BACalendarMetric_constrainedness_bucket",
                        "filename_base": "OverallMetrics_WorstOfN_by_constrainedness",
                        "normalize": True,
                        "agg_fn": "min",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "worstofn_eval_report"),
        )

        # Aggregate the results by a majority vote
        self.maj_vote_data_post_processing = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.evalreporting_comp.output_dir, "metric_results.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(
                                name_mapping={
                                    "model_output": "model_output_onerun",
                                }
                            ),
                            AddColumn("model_output"),
                            MajorityVoteTransform(model_output_col="model_output_onerun"),
                            CopyColumn("majority_vote", "model_output"),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_majvote_output"),
        )
        # Second, compute exact match
        self.majvote_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.maj_vote_data_post_processing.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            AddColumnAndData("count", 1),
                            RunPythonTransform("df = df[df['data_repeat_id'] == 'repeat_0']"),
                        ]
                    ),
                },
            ),
            metric_config=MetricConfig(BACalendarMetric),
            aggregator_configs=[
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": [
                            "BACalendarMetric_all_correct",
                            "BACalendarMetric_fraction_passed",
                            "BACalendarMetric_availability_programmatic_check",
                            "BACalendarMetric_meeting_duration_programmatic_check",
                            "BACalendarMetric_buffer_time_programmatic_check",
                            "BACalendarMetric_no_weekends_programmatic_check",
                            "BACalendarMetric_time_restrictions_programmatic_check",
                            "BACalendarMetric_specific_times_programmatic_check",
                            "BACalendarMetric_priority_programmatic_check",
                        ],
                        "filename_base": "OverallMetrics_MajVote",
                    },
                ),
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": [
                            "BACalendarMetric_all_correct",
                            "BACalendarMetric_fraction_passed",
                            "BACalendarMetric_availability_programmatic_check",
                            "BACalendarMetric_meeting_duration_programmatic_check",
                            "BACalendarMetric_buffer_time_programmatic_check",
                            "BACalendarMetric_no_weekends_programmatic_check",
                            "BACalendarMetric_time_restrictions_programmatic_check",
                            "BACalendarMetric_specific_times_programmatic_check",
                            "BACalendarMetric_priority_programmatic_check",
                        ],
                        "filename_base": "OverallMetrics_MajVote_by_constrainedness",
                        "group_by": "BACalendarMetric_constrainedness_bucket",
                    },
                ),
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": [
                            "count",
                        ],
                        "group_by": "BACalendarMetric_constrainedness_bucket",
                        "filename_base": "NumExamples_by_constrainedness",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "majvote_eval_report"),
        )

        # Configure the pipeline
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.evalreporting_comp,
                self.bon_evalreporting_comp,
                self.won_evalreporting_comp,
                self.maj_vote_data_post_processing,
                self.majvote_evalreporting_comp,
            ],
            self.log_dir,
        )


class BA_Calendar_Parallel_PIPELINE(BA_Calendar_PIPELINE):
    """Specifies the configuration for running the BA Calendar benchmark repeated 5 times.

    BA_Calendar_Parallel_PIPELINE extends BA_Calendar_PIPELINE with an adjusted
    data processing transform that multiplies the data by 5 repeats.
    """

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        """Configures the pipeline components and returns a PipelineConfig object.

        This method modifies the last transform step to multiply the dataset
        by 5.

        Args:
            model_config (ModelConfig): The model configuration object.
            resume_from (Optional[str]): Directory to resume from. Defaults to None.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            PipelineConfig: The configured pipeline with data repeated 5 times.
        """
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms[-1] = MultiplyTransform(
            n_repeats=5
        )
        return pipeline


class BA_Calendar_RunEvals_PIPELINE(BA_Calendar_PIPELINE):
    """Specifies the configuration for running BA Calendar benchmark.

    BA_Calendar_RunEvals_PIPELINE extends BA_Calendar_PIPELINE, focusing primarily
    on the evaluation steps. It adjusts the relevant data reader paths and
    can optionally sample the dataset.
    """

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, resume_logdir: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        """Configures the pipeline components for evaluation and returns a PipelineConfig object.

        This method updates the path for reading inference results and
        sets up the evaluation and reporting components.

        Args:
            model_config (ModelConfig): The model configuration object.
            resume_from (Optional[str]): The path to resume from. Defaults to None.
            resume_logdir (Optional[str]): The path to the log directory. Defaults to None.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            PipelineConfig: The pipeline configured for evaluation steps only.
        """
        pipeline = super().configure_pipeline(
            model_config=model_config, resume_from=resume_from, resume_logdir=resume_logdir
        )
        self.evalreporting_comp.data_reader_config.init_args["path"] = resume_from
        # self.data_processing_comp.data_reader_config.init_args["transform"].transforms.insert(0, SamplerTransform(random_seed=5, sample_count=100))
        # self.maj_vote_data_post_processing.data_reader_config.init_args["transform"].transforms.insert(0, SamplerTransform(random_seed=5, sample_count=100))
        # self.evalreporting_comp.data_reader_config.init_args["transform"].transforms.insert(0, SamplerTransform(random_seed=5, sample_count=100))
        # # self.bon_evalreporting_comp.data_reader_config.init_args["transform"].transforms.insert(0, SamplerTransform(random_seed=5, sample_count=100))
        # self.majvote_evalreporting_comp.data_reader_config.init_args["transform"].transforms.insert(0, SamplerTransform(random_seed=5, sample_count=100))
        return PipelineConfig(
            [
                self.evalreporting_comp,
                self.bon_evalreporting_comp,
                self.won_evalreporting_comp,
                self.maj_vote_data_post_processing,
                self.majvote_evalreporting_comp,
            ],
            self.log_dir,
        )
