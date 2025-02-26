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
    AddColumn,
    ColumnRename,
    DataReader,
    HFDataReader,
    MajorityVoteTransform,
    MMDataLoader,
    MultiplyTransform,
    SequenceTransform,
    ExtractUsageTransform,
    CopyColumn,
    ReplaceStringsTransform    
)
from eureka_ml_insights.data_utils.nphard_sat_utils import (
    NPHARDSATExtractAnswer,
)
from eureka_ml_insights.metrics import CountAggregator, NPHardSATMetric, BiLevelAggregator, BiLevelCountAggregator

"""This file contains user defined configuration classes for the Traveling Salesman Problem (SAT).
"""


class NPHARD_SAT_PIPELINE(ExperimentConfig):
    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        # Configure the data processing component.
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "GeoMeterData/nphard_sat1",
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
                # os.path.dirname(__file__), "../prompt_templates/nphard_sat_templates/Template_sat_o1.jinja"
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
                            NPHARDSATExtractAnswer("raw_output", "model_output"),
                            ExtractUsageTransform(model_config),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_post_processing_output"),
        )

###################################################

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
                AggregatorConfig(CountAggregator, 
                {
                    "column_names": ["NPHardSATMetric_result"],
                    "group_by": "data_repeat_id",
                    "filename_base": "NPHardSATMetric_result_SeparateRuns",
                    "normalize": True,
                }),
                AggregatorConfig(CountAggregator, 
                {
                    "column_names": ["NPHardSATMetric_result"],
                    "group_by": ["data_repeat_id", "category"],
                    "filename_base": "NPHardSATMetric_GroupBy_Category_SeparateRuns",
                    "normalize": True,
                }),    
                # the next two reports take the average and std for all repeats
                # the resulting numbers are the average and std of N pass@1 scores, where N is number of repeats
                AggregatorConfig(BiLevelCountAggregator, 
                {
                    "column_names": ["NPHardSATMetric_result"], 
                    "first_groupby": "data_repeat_id", 
                    "filename_base": "NPHardSATMetric_AllRuns",
                    "normalize": True
                }),
                AggregatorConfig(BiLevelCountAggregator, 
                {
                    "column_names": ["NPHardSATMetric_result"], 
                    "first_groupby": ["data_repeat_id", "category"], 
                    "second_groupby": "category",
                    "filename_base": "NPHardSATMetric_GroupBy_Category_AllRuns", 
                    "normalize": True
                }),     

                # the next two reports take the average and std for all repeats
                # for generating category and 
                # the resulting numbers are the average and std of N pass@1 scores, where N is number of repeats
                AggregatorConfig(BiLevelCountAggregator, 
                {
                    "column_names": ["NPHardSATMetric_result"], 
                    "first_groupby": "data_repeat_id", 
                    "filename_base": "NPHardSATMetric_AllRuns",
                    "normalize": True
                }),
                AggregatorConfig(BiLevelCountAggregator,
                {
                    "column_names": ["NPHardSATMetric_result"], 
                    "first_groupby": ["data_repeat_id", "category", "num_var"],
                    "second_groupby": ["category", "num_var"],
                    "filename_base": "NPHardSATMetric_GroupBy_Category_num_var_AllRuns",
                    "normalize": True
                }),

                # # two similar reports for average completion usage
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
                    "first_groupby": ["data_point_id", "category"], 
                    "second_groupby": "category",
                    "filename_base": "UsageCompletion_GroupBy_Category_AllRuns", 
                    "agg_fn": "mean"
                }),                                                     
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

#############################################

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
                            NPHARDSATExtractAnswer("raw_output", "model_output"),
                            MajorityVoteTransform(id_col="data_point_id"),
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
            metric_config=MetricConfig(NPHardSATMetric),
            aggregator_configs=[
                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": [
                            "NPHardSATMetric_result",
                        ],
                        "first_groupby": "data_point_id",
                        "filename_base": "MajorityVote",
                        "normalize": True,
                    },
                ),         
            ],
            output_dir=os.path.join(self.log_dir, "majorityvote_eval_report"),
        )


#############################################


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
                                column_name_src="NPHardSATMetric_result",
                                column_name_dst="NPHardSATMetric_result_numeric",
                            ),
                        ReplaceStringsTransform(
                                columns=["NPHardSATMetric_result_numeric"],
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
                            "NPHardSATMetric_result_numeric"
                        ],
                        "first_groupby": "data_point_id",
                        "filename_base": "NPHardSATMetric_BestOfN",
                        "agg_fn": "max"
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "NPHardSATMetric_result_numeric"
                        ],
                        "first_groupby": "data_point_id", 
                        "second_groupby": "category",
                        "filename_base": "NPHardSATMetric_BestOfN_GroupBy_Category",
                        "agg_fn": "max"
                    },
                ),
                # # # aggregates results by data_point_id and takes the sum of usage for completion tokens
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
###############################################
############## worst of n ######################

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
                            "NPHardSATMetric_result_numeric"
                        ],
                        "first_groupby": "data_point_id",
                        "filename_base": "NPHardSATMetric_WorstOfN",
                        "agg_fn": "min"
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "NPHardSATMetric_result_numeric"
                        ],
                        "first_groupby": "data_point_id", 
                        "second_groupby": "category",
                        "filename_base": "NPHardSATMetric_WorstOfN_GroupBy_Category",
                        "agg_fn": "min"
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "worstofn_eval_report"),
        )

###############################################

        # Configure the pipeline
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.data_post_processing,
                self.evalreporting_comp,
                self.data_post_processing_addmv,
                self.mv_evalreporting_comp,
                self.posteval_data_post_processing_comp,
                # self.bon_evalreporting_comp,
                # self.won_evalreporting_comp   
            ],
            self.log_dir,
        )
####################################

class NPHARD_SAT_PIPELINE_MULTIPLE_RUNS(NPHARD_SAT_PIPELINE):
    """This class specifies the config for running SAT benchmark n repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=5)
        )
        return pipeline


##################


        # # Configure the evaluation and reporting component.
        # self.evalreporting_comp = EvalReportingConfig(
        #     component_type=EvalReporting,
        #     data_reader_config=DataSetConfig(
        #         DataReader,
        #         {
        #             "path": os.path.join(self.data_post_processing.output_dir, "transformed_data.jsonl"),                    
        #             "format": ".jsonl",
        #         },
        #     ),
        #     metric_config=MetricConfig(NPHardSATMetric),
        #     aggregator_configs=[
        #         AggregatorConfig(CountAggregator, {"column_names": ["NPHardSATMetric_result"], "normalize": True}),
        #     ],
        #     output_dir=os.path.join(self.log_dir, "eval_report"),
        # )

        # # Aggregate the results by a majority vote.
        # self.postevalprocess_comp = EvalReportingConfig(
        #     component_type=EvalReporting,
        #     data_reader_config=DataSetConfig(
        #         DataReader,
        #         {
        #             "path": os.path.join(self.data_post_processing.output_dir, "transformed_data.jsonl"),
        #             "format": ".jsonl",
        #             "transform": SequenceTransform(
        #                 [
        #                     MajorityVoteTransform(id_col="data_point_id"),
        #                     ColumnRename(
        #                         name_mapping={
        #                             "model_output": "model_output_onerun",
        #                             "majority_vote": "model_output",
        #                         }
        #                     ),
        #                 ]
        #             ),
        #         },
        #     ),
        #     metric_config=MetricConfig(NPHardSATMetric),
        #     aggregator_configs=[
        #         AggregatorConfig(CountAggregator, {"column_names": ["NPHardSATMetric_result"], "normalize": True}),
        #     ],
        #     output_dir=os.path.join(self.log_dir, "eval_report_majorityVote"),
        # )