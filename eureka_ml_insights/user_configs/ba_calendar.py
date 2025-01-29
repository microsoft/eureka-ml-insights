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
    ColumnRename,
    CopyColumn,
    MajorityVoteTransform,
    MultiplyTransform,
    RunPythonTransform,
    SequenceTransform,
)
from eureka_ml_insights.metrics.ba_calendar_metrics import BACalendarMetric
from eureka_ml_insights.metrics.reports import (
    AverageAggregator,
    BiLevelAggregator,
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
    """This class specifies the config for running any benchmark on any model"""

    def configure_pipeline(self, model_config=None, resume_from=None, **kwargs) -> PipelineConfig:
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
                        "filename_base": "BaCal_OverallMetrics_SeparateRuns",
                        "group_by": "data_repeat_id",
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
                        "filename_base": "BaCal_BestOfN_Aggregated",
                        "normalize": True,
                        "agg_fn": "max",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "bestofn_eval_report"),
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
        # Second, compute eaxct match
        self.majvote_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.maj_vote_data_post_processing.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
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
                        "filename_base": "BaCal_MajVote_OverallMetrics_Aggregated",
                        "group_by": "data_repeat_id",
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
                self.maj_vote_data_post_processing,
                self.majvote_evalreporting_comp,
            ],
            self.log_dir,
        )


class BA_Calendar_Parallel_PIPELINE(BA_Calendar_PIPELINE):
    """This class specifies the config for running BA Calendar benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms[-1] = MultiplyTransform(
            n_repeats=5
        )
        return pipeline
