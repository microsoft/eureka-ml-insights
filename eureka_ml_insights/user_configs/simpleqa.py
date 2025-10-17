import os
from typing import Any

from eureka_ml_insights.core import (
    Inference,
    PromptProcessing,
)

from eureka_ml_insights.core.data_processing import DataProcessing
from eureka_ml_insights.core.eval_reporting import EvalReporting
from eureka_ml_insights.data_utils.ba_calendar_utils import BA_Calendar_ExtractAnswer
from eureka_ml_insights.data_utils.data import (
    DataLoader,
    DataReader,
    HFDataReader,
)
from eureka_ml_insights.data_utils.omni_math_utils import Omni_Math_ParseLabel, Omni_Math_ParseSolution
from eureka_ml_insights.data_utils.transform import AddColumn, AddColumnAndData, ColumnRename, CopyColumn, ExtractUsageTransform, MajorityVoteTransform, MultiplyTransform, ReplaceStringsTransform, RunPythonTransform, SamplerTransform, SequenceTransform
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
    PipelineConfig,
    PromptProcessingConfig,
    ModelConfig,
)
from ..configs.experiment_config import ExperimentConfig
from ..metrics import SimpleQA_Metric
from ..metrics.simpleqa_metrics import SQA_CGAAggregator, SQA_CGAAvgPass1Aggregator
from ..data_utils.simpleqa_utils import SimpleQA_MetadataExplode

class SimpleQA_PIPELINE(ExperimentConfig):
    """This class specifies the config for running SimpleQA benchmark on any model"""

    def configure_pipeline(self, model_config=None, resume_from=None, eval_resume_from=None, eval_model_config=None, **kwargs) -> PipelineConfig:
        # data preprocessing

        self.data_processing_comp = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                   "path": "lighteval/SimpleQA",
                   "split": "test",
                   "transform": SequenceTransform([
                    SamplerTransform(sample_count=100, random_seed=42),
                    MultiplyTransform(n_repeats=int(kwargs.get("n_repeats", 1))),
                    ColumnRename(name_mapping={"problem":"prompt", "answer":"ground_truth"}),
                    SimpleQA_MetadataExplode(metadata_column="metadata"),
                   ]),
                }
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        # inference component
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                DataLoader,
                {
                    "path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl"),
                },
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
            max_concurrent=int(kwargs.get("max_concurrent", 10)),
        )

        # eval data preprocessing
        self.eval_data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            prompt_template_path=os.path.join(os.path.dirname(__file__), "../prompt_templates/simpleqa_templates/simpleqa_grader_prompt.jinja"),
            data_reader_config=DataSetConfig(
                DataReader,
                {"path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                 "transform": SequenceTransform([
                    AddColumn("generated_solution"),
                    AddColumn("gen_solution_n_output_tokens"),
                    AddColumn("gen_solution_usage"),
                    AddColumn("gen_solution_is_valid"),
                    CopyColumn("model_output", "generated_solution"),
                    ColumnRename(name_mapping={"n_output_tokens":"gen_solution_n_output_tokens",
                                               "usage": "gen_solution_usage",
                                               "is_valid": "gen_solution_is_valid"}),
                 ])},
            ),
            output_dir=os.path.join(self.log_dir, "eval_data_processing_output"),
        )

        # inference component
        self.eval_inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=eval_model_config,
            data_loader_config=DataSetConfig(
                DataLoader,
                {"path": os.path.join(self.eval_data_processing_comp.output_dir, "transformed_data.jsonl")
                },
            ),
            output_dir=os.path.join(self.log_dir, "eval_inference_result"),
            resume_from=eval_resume_from,
            max_concurrent=40,
        )

        # Configure the evaluation and reporting component for evaluation and dataset level aggregation
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.eval_inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            ExtractUsageTransform(model_config, usage_column="gen_solution_usage", n_tokens_column="gen_solution_n_output_tokens"),
                        ]
                    ),
                },
            ),
            metric_config=MetricConfig(SimpleQA_Metric),
            aggregator_configs=[
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": [
                            "SimpleQA_Metric_is_correct",
                            "SimpleQA_Metric_is_incorrect",
                            "SimpleQA_Metric_is_not_attempted",
                        ],
                        "filename_base": "Metrics_SeparateRuns",
                        "group_by": "data_repeat_id",
                    },
                ),
                AggregatorConfig(SQA_CGAAggregator,
                    {
                        "is_correct_column_name": "SimpleQA_Metric_is_correct",
                        "is_incorrect_column_name": "SimpleQA_Metric_is_incorrect",
                        "is_not_attempted_column_name": "SimpleQA_Metric_is_not_attempted",
                        "group_by": "data_repeat_id",
                        "filename_base": "AccuracyGivenAttempted_SeparateRuns", 
                    }
                ),
                
                # the next three reports take the average and std for all repeats
                # the resulting numbers are the average and std of N pass@1 scores, where N is number of repeats
                AggregatorConfig(BiLevelAggregator, 
                    {
                        "column_names": [
                            "SimpleQA_Metric_is_correct",
                            "SimpleQA_Metric_is_incorrect",
                            "SimpleQA_Metric_is_not_attempted"
                        ], 
                        "first_groupby": "data_repeat_id", 
                        "filename_base": "Metrics_Avg",
                        "agg_fn": "mean"
                    }),
                AggregatorConfig(BiLevelAggregator, 
                    {
                        "column_names": [
                            "SimpleQA_Metric_is_correct",
                            "SimpleQA_Metric_is_incorrect",
                            "SimpleQA_Metric_is_not_attempted"
                        ], 
                        "first_groupby": ["data_repeat_id", "topic"], 
                        "second_groupby": "topic",
                        "filename_base": "Metrics_Avg_by_topic",
                        "agg_fn": "mean"
                    }),
                
                AggregatorConfig(SQA_CGAAvgPass1Aggregator,
                    {
                        "is_correct_column_name": "SimpleQA_Metric_is_correct",
                        "is_incorrect_column_name": "SimpleQA_Metric_is_incorrect",
                        "is_not_attempted_column_name": "SimpleQA_Metric_is_not_attempted",
                        "filename_base": "AccuracyGivenAttempted_Avg", 
                    }
                ),
                AggregatorConfig(SQA_CGAAvgPass1Aggregator,
                    {
                        "is_correct_column_name": "SimpleQA_Metric_is_correct",
                        "is_incorrect_column_name": "SimpleQA_Metric_is_incorrect",
                        "is_not_attempted_column_name": "SimpleQA_Metric_is_not_attempted",
                        "filename_base": "AccuracyGivenAttempted_Avg_by_topic", 
                        "group_by": "topic",
                    }
                ), 

                # reports for average completion usage
                AggregatorConfig(BiLevelAggregator, 
                    {
                        "column_names": ["usage_completion"], 
                        "first_groupby": "data_point_id", 
                        "filename_base": "UsageCompletion",
                        "agg_fn": "mean"
                    }),
                AggregatorConfig(BiLevelAggregator, 
                    {
                        "column_names": ["usage_completion"], 
                        "first_groupby": ["data_point_id", "topic"],
                        "second_groupby": "topic",
                        "filename_base": "UsageCompletion_by_topic",
                        "agg_fn": "mean"
                    }),

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
                            "SimpleQA_Metric_is_correct",
                        ],
                        "first_groupby": "data_point_id",
                        "filename_base": "Correctness_BestofN",
                        "normalize": True,
                        "agg_fn": "max",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "SimpleQA_Metric_is_correct",
                        ],
                        "first_groupby": "data_point_id",
                        "second_groupby": "topic",
                        "filename_base": "Correctness_BestOfN_by_topic",
                        "normalize": True,
                        "agg_fn": "max",
                    },
                ),
                
                # aggregates results by data_point_id and takes the sum of usage for completion tokens
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "usage_completion"
                        ],
                        "first_groupby": "data_point_id",
                        "filename_base": "UsageCompletion_BestOfN",
                         "agg_fn": "sum"
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
                            "SimpleQA_Metric_is_correct",
                        ],
                        "first_groupby": "data_point_id",
                        "filename_base": "Correctness_WorstofN",
                        "normalize": True,
                        "agg_fn": "min",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "SimpleQA_Metric_is_correct",
                        ],
                        "first_groupby": "data_point_id",
                        "second_groupby": "topic",
                        "filename_base": "Correctness_WorstOfN_by_topic",
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
                                    "SimpleQA_Metric_grade": "SimpleQA_Metric_grade_onerun",
                                }
                            ),
                            AddColumn("SimpleQA_Metric_grade"),
                            MajorityVoteTransform(model_output_col="SimpleQA_Metric_grade_onerun", model_label_column="SimpleQA_Metric_is_correct"),
                            RunPythonTransform("df = df.rename_axis(index={'data_point_id': 'data_point_id_idx'})"),
                            CopyColumn("majority_label", "SimpleQA_Metric_is_correct_majority_vote"),
                            MajorityVoteTransform(model_output_col="SimpleQA_Metric_grade_onerun", model_label_column="SimpleQA_Metric_is_incorrect"),
                            RunPythonTransform("df = df.rename_axis(index={'data_point_id': 'data_point_id_idx'})"),
                            CopyColumn("majority_label", "SimpleQA_Metric_is_incorrect_majority_vote"),
                            MajorityVoteTransform(model_output_col="SimpleQA_Metric_grade_onerun", model_label_column="SimpleQA_Metric_is_not_attempted"),
                            RunPythonTransform("df = df.rename_axis(index={'data_point_id': 'data_point_id_idx'})"),
                            CopyColumn("majority_label", "SimpleQA_Metric_is_not_attempted_majority_vote"),
                            CopyColumn("majority_vote", "SimpleQA_Metric_majority_vote_grade"),
                            RunPythonTransform("df = df.drop(columns=['SimpleQA_Metric_grade_onerun', 'majority_label', 'majority_vote'])"),
                            AddColumnAndData("count", 1),

                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_majvote_output"),
        )

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
            aggregator_configs=[
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": [
                            "SimpleQA_Metric_is_correct_majority_vote",
                            "SimpleQA_Metric_is_incorrect_majority_vote",
                            "SimpleQA_Metric_is_not_attempted_majority_vote"
                        ],
                        "filename_base": "Correctness_MajVote",
                    },
                ),
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": [
                            "SimpleQA_Metric_is_correct_majority_vote",
                            "SimpleQA_Metric_is_incorrect_majority_vote",
                            "SimpleQA_Metric_is_not_attempted_majority_vote"
                        ],
                        "filename_base": "Correctness_MajVote_by_topic",
                        "group_by": "topic",
                    },
                ),
                AggregatorConfig(SQA_CGAAggregator,
                    {
                        "is_correct_column_name": "SimpleQA_Metric_is_correct_majority_vote",
                        "is_incorrect_column_name": "SimpleQA_Metric_is_incorrect_majority_vote",
                        "is_not_attempted_column_name": "SimpleQA_Metric_is_not_attempted_majority_vote",
                        "filename_base": "AccuracyGivenAttempted_MajVote", 
                    }
                ),
                AggregatorConfig(CountAggregator, 
                    {
                        "column_names": [
                            "count",
                        ], 
                        "group_by": "topic", 
                        "filename_base": "NumExamples_by_topic",
                    }),
            ],
            output_dir=os.path.join(self.log_dir, "majvote_eval_report"),
        )

        # Configure the pipeline
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.eval_data_processing_comp,
                self.eval_inference_comp,
                self.evalreporting_comp,
                self.bon_evalreporting_comp,
                self.won_evalreporting_comp,
                self.maj_vote_data_post_processing,
                self.majvote_evalreporting_comp,
            ],
            self.log_dir,
        )

class SimpleQA_Verified_PIPELINE(SimpleQA_PIPELINE):
    """This class specifies the config for running SimpleQA Verified benchmark on any model"""

    def configure_pipeline(self, model_config=None, resume_from=None, eval_resume_from=None, eval_model_config=None, **kwargs) -> PipelineConfig:
        # call the parent class method to get the base pipeline config
        pipeline_config = super().configure_pipeline(
            model_config=model_config,
            resume_from=resume_from,
            eval_resume_from=eval_resume_from,
            eval_model_config=eval_model_config,
            **kwargs
        )

        # modify the data processing component to use the Verified split
        self.data_processing_comp.data_reader_config.init_args["path"] = "google/simpleqa-verified"
        self.data_processing_comp.data_reader_config.init_args["split"] = "eval"
        num_transforms = len(self.data_processing_comp.data_reader_config.init_args["transform"].transforms)
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms = self.data_processing_comp.data_reader_config.init_args["transform"].transforms[0:num_transforms-1]  # remove last transform which is SimpleQA_MetadataExplode
        return pipeline_config