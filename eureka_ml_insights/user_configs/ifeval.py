import os
from typing import Any, Optional

from eureka_ml_insights.core import (
    DataProcessing,
    Inference,
    PromptProcessing,
)
from eureka_ml_insights.core.eval_reporting import EvalReporting
from eureka_ml_insights.data_utils import ColumnRename
from eureka_ml_insights.data_utils.data import (
    DataLoader,
    DataReader,
    HFDataReader,
)
from eureka_ml_insights.data_utils.transform import RunPythonTransform, SamplerTransform
from eureka_ml_insights.metrics.ifeval_metrics import IFEvalMetric
from eureka_ml_insights.metrics.reports import (
    AverageAggregator,
    TwoColumnSumAverageAggregator,
)

from eureka_ml_insights.configs import(
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
from eureka_ml_insights.configs import ExperimentConfig


class IFEval_PIPELINE(ExperimentConfig):
    """This class specifies the config for running IFEval benchmark on any model"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, 
        **kwargs: dict[str, Any]) -> PipelineConfig:

        # data preprocessing
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "google/IFEval",
                    "split": "train",
                    # "transform": SamplerTransform(sample_count=20, random_seed=99),
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
        )

        # Configure the evaluation and reporting component for evaluation and dataset level aggregation
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": ColumnRename(name_mapping={"model_output": "response"}),
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
                        "filename_base": "IFEvalAccuracyMetrics_Aggregated",
                    },
                ),
                AggregatorConfig(
                    TwoColumnSumAverageAggregator,
                    {
                        "numerator_column_name": "IFEvalMetric_strict_follow_instruction_list_sum",
                        "denominator_column_name": "IFEvalMetric_strict_instruction_list_len",
                        "filename_base": "IFEvalStrictInfoFollowRateMetric_Aggregated",
                    },
                ),
                AggregatorConfig(
                    TwoColumnSumAverageAggregator,
                    {
                        "numerator_column_name": "IFEvalMetric_loose_follow_instruction_list_sum",
                        "denominator_column_name": "IFEvalMetric_loose_instruction_list_len",
                        "filename_base": "IFEvalLooseInfoFollowRateMetric_Aggregated",
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
                    AverageAggregator,
                    {
                        "column_names": [
                            "IFEvalMetric_strict_follow_instruction_list",
                            "IFEvalMetric_loose_follow_instruction_list",
                        ],
                        "group_by": "instruction_id_list",
                        "filename_base": "IFEvalAccuracyMetrics_GroupByInstructionID",
                    },
                ),
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": [
                            "IFEvalMetric_strict_follow_instruction_list",
                            "IFEvalMetric_loose_follow_instruction_list",
                        ],
                        "group_by": "IFEvalMetric_tier0_instructions",
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
