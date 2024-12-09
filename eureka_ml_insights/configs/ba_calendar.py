import os
from tkinter import N

from eureka_ml_insights.core import (
    Inference,
    PromptProcessing,
)

from eureka_ml_insights.core.eval_reporting import EvalReporting
from eureka_ml_insights.data_utils.data import (
    DataLoader,
    DataReader,
    HFDataReader,
)
from eureka_ml_insights.data_utils.transform import ColumnRename, SamplerTransform, SequenceTransform
from eureka_ml_insights.metrics.ba_calendar_metrics import BACalendarMetric
from eureka_ml_insights.metrics.reports import (
    AverageAggregator,
    NAFilteredAverageAggregator,
)

from .config import (
    AggregatorConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    MetricConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from .experiment_config import ExperimentConfig

class Calendar_Schedule_PIPELINE(ExperimentConfig):
    """This class specifies the config for running any benchmark on any model"""

    def configure_pipeline(self, model_config=None, resume_from=None, **kwargs) -> PipelineConfig:
        # data preprocessing
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            prompt_template_path=os.path.join(os.path.dirname(__file__), "../prompt_templates/ba_calendar_templates/calendar_scheduling.jinja"),
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "microsoft/ba-calendar",
                    "split": "test",
                    "transform": SequenceTransform([
                        ColumnRename(name_mapping={"task_prompt": "prompt"}),
                    ]),
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
            # max_concurrent=4,
        )

        # Configure the evaluation and reporting component for evaluation and dataset level aggregation
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(BACalendarMetric),
            aggregator_configs=[
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": [
                            "BACalendarMetric_all_correct",
                            "BACalendarMetric_fraction_passed"
                        ],
                        "filename_base": "BaCal_OverallMetrics_Aggregated",
                    },
                ),
                AggregatorConfig(
                    NAFilteredAverageAggregator,
                    {
                        "column_name": "BACalendarMetric_availability_programmatic_check",
                        "filename_base": "BaCal_Availability_Check_Aggregated",
                    },
                ),
                AggregatorConfig(
                    NAFilteredAverageAggregator,
                    {
                        "column_name": "BACalendarMetric_meeting_duration_programmatic_check",
                        "filename_base": "BaCal_MeetingDuration_Check_Aggregated",
                    },
                ),
                AggregatorConfig(
                    NAFilteredAverageAggregator,
                    {
                        "column_name": "BACalendarMetric_buffer_time_programmatic_check",
                        "filename_base": "BaCal_BufferTime_Check_Aggregated",
                    },
                ),
                AggregatorConfig(
                    NAFilteredAverageAggregator,
                    {
                        "column_name": "BACalendarMetric_no_weekends_programmatic_check",
                        "filename_base": "BaCal_NoWeekends_Check_Aggregated",
                    },
                ),
                AggregatorConfig(
                    NAFilteredAverageAggregator,
                    {
                        "column_name": "BACalendarMetric_time_restrictions_programmatic_check",
                        "filename_base": "BaCal_TimeRestrictions_Check_Aggregated",
                    },
                ),
                AggregatorConfig(
                    NAFilteredAverageAggregator,
                    {
                        "column_name": "BACalendarMetric_specific_times_programmatic_check",
                        "filename_base": "BaCal_SpecificTimes_Check_Aggregated",
                    },
                ),
                AggregatorConfig(
                    NAFilteredAverageAggregator,
                    {
                        "column_name": "BACalendarMetric_priority_programmatic_check",
                        "filename_base": "BaCal_Priority_Check_Aggregated",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # Configure the pipeline
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.evalreporting_comp
            ],
            self.log_dir,
        )