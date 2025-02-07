import os
from typing import Any

from eureka_ml_insights.configs import (
    AggregatorConfig,
    DataJoinConfig,
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
from eureka_ml_insights.core import DataProcessing, Inference, PromptProcessing, DataJoin
from eureka_ml_insights.core.eval_reporting import EvalReporting
from eureka_ml_insights.data_utils import (
    AddColumn,
    ColumnRename,
    DataReader,
    HFDataReader,
    MajorityVoteTransform,
    MultiplyTransform,
    SequenceTransform,
    ExtractUsageTransform,
    CopyColumn,
    ReplaceStringsTransform
)
from eureka_ml_insights.data_utils.aime_utils import AIMEExtractAnswer
from eureka_ml_insights.data_utils.data import DataLoader
from eureka_ml_insights.metrics.aime_metrics import NumericMatch
from eureka_ml_insights.metrics.reports import (
    BiLevelCountAggregator,
    CountAggregator,
    BiLevelAggregator
)

# from eureka_ml_insights.data_utils.transform import MajorityVoteTransform


class AIME_PIPELINE(ExperimentConfig):
    """This class specifies the config for running AIME benchmark on any model"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:

        # data preprocessing
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "qq8933/AIME_1983_2024",
                    "split": "train",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(
                                name_mapping={
                                    "Question": "prompt",
                                    "Answer": "ground_truth",
                                }
                            ),
                        ],
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/aime_templates/Template_1clean.jinja"
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
                            ColumnRename(
                                name_mapping={
                                    "model_output": "raw_output",
                                }
                            ),
                            AddColumn("model_output"),
                            AIMEExtractAnswer("raw_output", "model_output"),
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
            metric_config=MetricConfig(NumericMatch),
            aggregator_configs=[
                # the first two reports aggregate the metrics per experiment repeat
                # each repeat can be considered as an individual pass@1 score
                AggregatorConfig(CountAggregator, 
                {
                    "column_names": ["NumericMatch_result"], 
                    "group_by": "data_repeat_id", 
                    "filename_base": "NumericMatch_SeparateRuns",
                    "normalize": True
                }),
                AggregatorConfig(CountAggregator, 
                {
                    "column_names": ["NumericMatch_result"], 
                    "group_by": ["data_repeat_id", "Year"], 
                    "filename_base": "NumericMatch_GroupBy_Year_SeparateRuns", 
                    "normalize": True
                }),
                # the next two reports take the average and std for all repeats
                # the resulting numbers are the average and std of N pass@1 scores, where N is number of repeats
                AggregatorConfig(BiLevelCountAggregator, 
                {
                    "column_names": ["NumericMatch_result"], 
                    "first_groupby": "data_repeat_id", 
                    "filename_base": "NumericMatch_AllRuns",
                    "normalize": True
                }),
                AggregatorConfig(BiLevelCountAggregator, 
                {
                    "column_names": ["NumericMatch_result"], 
                    "first_groupby": ["data_repeat_id",    "Year"], 
                    "second_groupby": "Year",
                    "filename_base": "NumericMatch_GroupBy_Year_AllRuns", 
                    "normalize": True
                }),
                # two similar reports for average completion usage
                AggregatorConfig(BiLevelAggregator, 
                {
                    "column_names": ["usage_completion"], 
                    "first_groupby": "data_point_id", 
                    "filename_base": "UsageCompletion_AllRuns",
                    "agg_fn": "mean"
                }),
                AggregatorConfig(BiLevelAggregator, 
                {
                    "column_names": ["usage_completion"], 
                    "first_groupby": ["data_point_id", "Year"], 
                    "second_groupby": "Year",
                    "filename_base": "UsageCompletion_GroupBy_Year_AllRuns", 
                    "agg_fn": "mean"
                }),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # Aggregate the results by a majority vote
        # First, let us perform majority_vote
        self.data_post_processing_addmv = DataProcessingConfig(
            component_type=DataProcessing,
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
                            AIMEExtractAnswer("raw_output", "model_output"),
                            MajorityVoteTransform(id_col="ID"),
                            ColumnRename(
                                name_mapping={
                                    "model_output": "model_output_onerun",
                                    "majority_vote": "model_output",
                                }
                            ),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_addmv_output"),
        )
        # Second, compute numeric match
        self.mv_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.data_post_processing_addmv.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(NumericMatch),
            aggregator_configs=[
                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": [
                            "NumericMatch_result",
                        ],
                        "first_groupby": "ID",
                        "filename_base": "MajorityVote",
                        "normalize": True,
                    },
                ),         
            ],
            output_dir=os.path.join(self.log_dir, "majorityvote_eval_report"),
        )

        self.posteval_data_post_processing_comp = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.evalreporting_comp.output_dir, "metric_results.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                        CopyColumn(
                                column_name_src="NumericMatch_result",
                                column_name_dst="NumericMatch_result_numeric",
                            ),
                        ReplaceStringsTransform(
                                columns=["NumericMatch_result_numeric"],
                                mapping={'incorrect': '0', 'correct': '1', 'none': 'NaN'},
                                case=False)
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "posteval_data_post_processing_output"),
        )
        # Aggregate the results by best of n
        # In this case, this is equivalent to taking the max on the numerical column of the metric.
        self.bon_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.posteval_data_post_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl"
                },
            ),
            aggregator_configs=[
                # the first three reports aggregate results by data_point_id and take the best out of N
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "NumericMatch_result_numeric"
                        ],
                        "first_groupby": "data_point_id",
                        "filename_base": "NumericMatch_BestOfN",
                        "agg_fn": "max"
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "NumericMatch_result_numeric"
                        ],
                        "first_groupby": "data_point_id", 
                        "second_groupby": "Year",
                        "filename_base": "NumericMatch_BestOfN_GroupBy_Year",
                        "agg_fn": "max"
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

        self.won_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.posteval_data_post_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl"
                },
            ),
            aggregator_configs=[
                # the first three reports aggregate results by data_point_id and take the best out of N
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "NumericMatch_result_numeric"
                        ],
                        "first_groupby": "data_point_id",
                        "filename_base": "NumericMatch_WorstOfN",
                        "agg_fn": "min"
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "NumericMatch_result_numeric"
                        ],
                        "first_groupby": "data_point_id", 
                        "second_groupby": "Year",
                        "filename_base": "NumericMatch_WorstOfN_GroupBy_Year",
                        "agg_fn": "min"
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "worstofn_eval_report"),
        )

        # Configure the pipeline
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.data_post_processing,
                self.evalreporting_comp,
                self.data_post_processing_addmv,
                self.mv_evalreporting_comp ,
                self.posteval_data_post_processing_comp,
                self.bon_evalreporting_comp,
                self.won_evalreporting_comp
            ],
            self.log_dir,
        )


class AIME_PIPELINE5Run(AIME_PIPELINE):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=5)
        )
        return pipeline

class AIME_PIPELINE5Run_2025(AIME_PIPELINE):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        self.data_processing_comp.data_reader_config = DataSetConfig(
                DataReader,
                {
                    # read from local file for 2025
                    "path": r"C:\Users\lingjiaochen\Downloads\AIME2025.csv",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(
                                name_mapping={
                                    "Question": "prompt",
                                    "Answer": "ground_truth",
                                }
                            ),
                        ],
                    ),
                },
            )
        # join the other answer
        answer_path = r"C:\Users\lingjiaochen\Downloads\AIME2025_Answer.csv"
        other_data_reader_config  = DataSetConfig(
                DataReader,
                {
                    "path": answer_path,
                     "transform": SequenceTransform(
                        [
                            ColumnRename(
                                name_mapping={
                                     "Answer_Label":"ground_truth",
                                }
                            ),
                        ]
                    ),
                },
            )
        # post process the response to extract the answer
        self.data_post_processing = DataJoinConfig(
            component_type=DataJoin,
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
                                    "ground_truth":"old_ground_truth",
                                }
                            ),
                            AddColumn("model_output"),
                            AIMEExtractAnswer("raw_output", "model_output"),
                            ExtractUsageTransform(model_config),
                        ]
                    ),
                },
            ),
            other_data_reader_config  = other_data_reader_config,
            pandas_merge_args ={"on":"ID"},
            output_dir=os.path.join(self.log_dir, "data_post_processing_output"),
        )
        
        # post process the response to extract the answer for majority vote
        self.data_post_processing_addmv = DataJoinConfig(
            component_type=DataJoin,
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
                                    "ground_truth":"old_ground_truth",

                                }
                            ),
                            AddColumn("model_output"),
                            AIMEExtractAnswer("raw_output", "model_output"),
                            MajorityVoteTransform(id_col="ID"),
                            ColumnRename(
                                name_mapping={
                                    "model_output": "model_output_onerun",
                                    "majority_vote": "model_output",
                                }
                            ),
                        ]
                    ),
                },
            ),
            other_data_reader_config  = other_data_reader_config,
            pandas_merge_args ={"on":"ID"},
            output_dir=os.path.join(self.log_dir, "data_addmv_output"),
        )
        

        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=5)
        )
        
        # Configure the pipeline; this is necessary for resume_from to work
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.data_post_processing,
                self.evalreporting_comp,
                self.data_post_processing_addmv,
                self.mv_evalreporting_comp ,
                self.posteval_data_post_processing_comp,
                self.bon_evalreporting_comp,
                self.won_evalreporting_comp
            ],
            self.log_dir,
        )


class AIME_PIPELINE16Run(AIME_PIPELINE):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=16)
        )
        return pipeline


class AIME_PIPELINE32Run(AIME_PIPELINE):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=32)
        )
        return pipeline


class AIME_PIPELINE64Run(AIME_PIPELINE):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=64)
        )
        return pipeline


class AIME_PIPELINE128Run(AIME_PIPELINE):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=128)
        )
        return pipeline


class AIME_PIPELINE256Run(AIME_PIPELINE):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=256)
        )
        return pipeline


class AIME_PIPELINE512Run(AIME_PIPELINE):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=512)
        )
        return pipeline


class AIME_PIPELINE1024Run(AIME_PIPELINE):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=1024)
        )
        return pipeline
