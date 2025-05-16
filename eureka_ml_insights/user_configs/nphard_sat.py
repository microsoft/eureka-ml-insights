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
from eureka_ml_insights.core import (
    DataProcessing,
    EvalReporting,
    Inference,
    PromptProcessing,
)
from eureka_ml_insights.data_utils import (
    ColumnRename,
    DataReader,
    ExtractUsageTransform,
    HFDataReader,
    MMDataLoader,
    MultiplyTransform,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.nphard_sat_utils import (
    NPHARDSATExtractAnswer,
)
from eureka_ml_insights.metrics import (
    BiLevelAggregator,
    BiLevelCountAggregator,
    CountAggregator,
    NPHardSATMetric,
)

"""This file contains user defined configuration classes for the SAT benchmark.
"""


class NPHARD_SAT_PIPELINE(ExperimentConfig):
    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, n_repeats: int = 1, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        # Configure the data processing component.
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "GeoMeterData/nphard_sat2",
                    "split": "train",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={"query_text": "prompt", "target_text": "ground_truth"}),
                        ]
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/nphard_sat_templates/Template_sat_cot.jinja"
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
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
            max_concurrent=5,
        )

        # post process the response to extract the answer
        self.data_post_processing = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            NPHARDSATExtractAnswer("model_output", "extracted_answer"),
                            ExtractUsageTransform(model_config),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_post_processing_output"),
        )

        # Configure the evaluation and reporting component for evaluation and dataset level aggregation
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.data_post_processing.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(NPHardSATMetric),
            aggregator_configs=[
                # the first two reports aggregate the metrics per experiment repeat
                # each repeat can be considered as an individual pass@1 score
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": ["NPHardSATMetric_result"],
                        "group_by": "data_repeat_id",
                        "filename_base": "NPHardSATMetric_result_SeparateRuns",
                        "normalize": True,
                    },
                ),
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": ["NPHardSATMetric_result"],
                        "group_by": ["data_repeat_id", "category"],
                        "filename_base": "NPHardSATMetric_GroupBy_Category_SeparateRuns",
                        "normalize": True,
                    },
                ),
                # the next two reports take the average and std for all repeats
                # the resulting numbers are the average and std of N pass@1 scores, where N is number of repeats
                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": ["NPHardSATMetric_result"],
                        "first_groupby": "data_repeat_id",
                        "filename_base": "NPHardSATMetric_AllRuns",
                        "normalize": True,
                    },
                ),
                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": ["NPHardSATMetric_result"],
                        "first_groupby": ["data_repeat_id", "category"],
                        "second_groupby": "category",
                        "filename_base": "NPHardSATMetric_GroupBy_Category_AllRuns",
                        "normalize": True,
                    },
                ),
                # the next two reports take the average and std for all repeats
                # for generating category and
                # the resulting numbers are the average and std of N pass@1 scores, where N is number of repeats
                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": ["NPHardSATMetric_result"],
                        "first_groupby": "data_repeat_id",
                        "filename_base": "NPHardSATMetric_AllRuns",
                        "normalize": True,
                    },
                ),
                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": ["NPHardSATMetric_result"],
                        "first_groupby": ["data_repeat_id", "category", "num_var"],
                        "second_groupby": ["category", "num_var"],
                        "filename_base": "NPHardSATMetric_GroupBy_Category_num_var_AllRuns",
                        "normalize": True,
                    },
                ),
                # # two similar reports for average completion usage
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
                        "first_groupby": ["data_point_id", "category"],
                        "second_groupby": "category",
                        "filename_base": "UsageCompletion_GroupBy_Category_AllRuns",
                        "agg_fn": "mean",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": ["data_point_id", "category", "num_var"],
                        "second_groupby": ["category", "num_var"],
                        "filename_base": "UsageCompletion_GroupBy_Category_num_var_AllRuns",
                        "agg_fn": "mean",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # Configure the pipeline
        return PipelineConfig(
            [self.data_processing_comp, self.inference_comp, self.data_post_processing, self.evalreporting_comp],
            self.log_dir,
        )


class NPHARD_SAT_PIPELINE_MULTIPLE_RUNS(NPHARD_SAT_PIPELINE):
    """This class specifies the config for running SAT benchmark n repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, n_repeats: int = 1, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=int(n_repeats))
        )
        return pipeline
