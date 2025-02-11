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
    CopyColumn,
    ReplaceStringsTransform
)
from eureka_ml_insights.data_utils.nphard_tsp_utils import (
    NPHARDTSPExtractAnswer,
)
from eureka_ml_insights.metrics import CountAggregator, NPHardTSPMetric, BiLevelAggregator, BiLevelCountAggregator

"""This file contains user defined configuration classes for the Traveling Salesman Problem (TSP).
"""


class NPHARD_TSP_PIPELINE(ExperimentConfig):
    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        # Configure the data processing component.
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "GeoMeterData/nphard_tsp2",
                    "split": "train",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={"query_text": "prompt", "target_text": "ground_truth"}),
                        ]
                    ),
                },
            ),
            prompt_template_path=os.path.join(                
                # os.path.dirname(__file__), "../prompt_templates/nphard_tsp_templates/Template_tsp_o1.jinja"
                os.path.dirname(__file__), "../prompt_templates/nphard_tsp_templates/Template_tsp_cot.jinja"
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
                            NPHARDTSPExtractAnswer("raw_output", "model_output"),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_post_processing_output"),
        )

        # Configure the evaluation and reporting component. 
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.data_post_processing.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(NPHardTSPMetric),
            aggregator_configs=[
                # the first two reports aggregate the metrics per experiment repeat
                # each repeat can be considered as an individual pass@1 score                
                AggregatorConfig(CountAggregator, 
                {
                    "column_names": ["NPHardTSPMetric_result"],
                    "group_by": "data_repeat_id",
                    "filename_base": "NPHardTSPMetric_result_SeparateRuns",
                    "normalize": True,
                }),
                AggregatorConfig(CountAggregator, 
                {
                    "column_names": ["NPHardTSPMetric_result"],
                    "group_by": ["data_repeat_id", "category"],
                    "filename_base": "NPHardTSPMetric_GroupBy_Year_SeparateRuns",
                    "normalize": True,
                }),    
                # the next two reports take the average and std for all repeats
                # the resulting numbers are the average and std of N pass@1 scores, where N is number of repeats
                AggregatorConfig(BiLevelCountAggregator, 
                {
                    "column_names": ["NPHardTSPMetric_result"], 
                    "first_groupby": "data_repeat_id", 
                    "filename_base": "NPHardTSPMetric_AllRuns",
                    "normalize": True
                }),
                AggregatorConfig(BiLevelCountAggregator, 
                {
                    "column_names": ["NPHardTSPMetric_result"], 
                    "first_groupby": ["data_repeat_id", "category"], 
                    "second_groupby": "category",
                    "filename_base": "NPHardTSPMetric_GroupBy_Year_AllRuns", 
                    "normalize": True
                }),       
                # # two similar reports for average completion usage
                # AggregatorConfig(BiLevelAggregator, 
                # {
                #     "column_names": ["usage", "completion_tokens"], 
                #     "first_groupby": "data_point_id", 
                #     "filename_base": "UsageCompletion_AllRuns",
                #     "agg_fn": "mean"
                # }),
                # AggregatorConfig(BiLevelAggregator, 
                # {
                #     "column_names": ["completion_tokens"], 
                #     "first_groupby": ["data_point_id", "category"], 
                #     "second_groupby": "category",
                #     "filename_base": "UsageCompletion_GroupBy_Year_AllRuns", 
                #     "agg_fn": "mean"
                # }),                                                     
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        #### Aggregate the results best of n.
        #### first convert correct to 1 and incorrect to 0, none to NaN

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
                                column_name_src="NPHardTSPMetric_result",
                                column_name_dst="NPHardTSPMetric_result_numeric",
                            ),
                        ReplaceStringsTransform(
                                columns=["NPHardTSPMetric_result_numeric"],
                                mapping={'incorrect': '0', 'correct': '1', 'none': 'NaN'},
                                case=False)
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "posteval_data_post_processing_output"),
        )


        # Aggregate the results best of n.
        self.bon_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.posteval_data_post_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),            
            aggregator_configs=[
                AggregatorConfig(BiLevelAggregator,
                                 {
                                     "column_names": ["NPHardTSPMetric_result_numeric"], 
                                     "normalize": True,
                                     "first_groupby": "data_point_id",
                                     "agg_fn": "max"
                                 }
                            ),
            ],
            output_dir=os.path.join(self.log_dir, "bestofn_eval_report"),
        )


        # Aggregate the results worst of n.
        self.won_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.posteval_data_post_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),            
            aggregator_configs=[
                AggregatorConfig(BiLevelAggregator,
                                 {
                                     "column_names": ["NPHardTSPMetric_result_numeric"], 
                                     "normalize": True,
                                     "first_groupby": "data_point_id",
                                     "agg_fn": "min"
                                 }
                            ),
            ],
            output_dir=os.path.join(self.log_dir, "worstofn_eval_report"),
        )

        # Aggregate the results by a majority vote.
        self.postevalprocess_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.data_post_processing.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
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
            metric_config=MetricConfig(NPHardTSPMetric),
            aggregator_configs=[
                AggregatorConfig(CountAggregator, {"column_names": ["NPHardTSPMetric_result"], "normalize": True}),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report_majorityVote"),
        )

        # Configure the pipeline
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.data_post_processing,
                self.evalreporting_comp,
                # self.posteval_data_post_processing_comp,
                # self.bon_evalreporting_comp,
                # self.won_evalreporting_comp,
                # self.postevalprocess_comp,
            ],
            self.log_dir,
        )


class NPHARD_TSP_PIPELINE_MULTIPLE_RUNS(NPHARD_TSP_PIPELINE):
    """This class specifies the config for running TSP benchmark n repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=2)
        )
        return pipeline
