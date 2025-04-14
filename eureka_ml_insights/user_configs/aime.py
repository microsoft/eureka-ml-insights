import os
from typing import Any

from eureka_ml_insights.core import (
    EvalReporting, 
    Inference, 
    PromptProcessing, 
    DataProcessing,
    DataJoin
)

from eureka_ml_insights.configs import (
    AggregatorConfig,
    DataProcessingConfig,
    DataSetConfig,
    EvalReportingConfig,
    ExperimentConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
DataJoinConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.core import DataProcessing, Inference, PromptProcessing
from eureka_ml_insights.core.eval_reporting import EvalReporting
from eureka_ml_insights.data_utils import (
    AddColumn,
    ColumnRename,
    DataReader,
    HFDataReader,
    MajorityVoteTransform,
    MultiplyTransform,
    SequenceTransform,
    SamplerTransform,
       CopyColumn,
       ImputeNA,
       ReplaceStringsTransform,
       ExtractUsageTransform,
       RunPythonTransform

)
from eureka_ml_insights.data_utils.aime_utils import AIMEExtractAnswer
from eureka_ml_insights.data_utils.data import DataLoader, MMDataLoader
from eureka_ml_insights.metrics.aime_metrics import NumericMatch

from eureka_ml_insights.metrics.reports import (
    BiLevelCountAggregator,
    CountAggregator,
    BiLevelAggregator,
)

from eureka_ml_insights.configs.model_configs import OAI_GPT4O_2024_11_20_CONFIG

# from eureka_ml_insights.data_utils.transform import MajorityVoteTransform


class AIME_PIPELINE(ExperimentConfig):
    """This class specifies the config for running AIME benchmark on any model"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        max_concurrent = kwargs.get("max_concurrent", 5)
        max_concurrent = int(max_concurrent)
        print(f"********max concurrent is {max_concurrent} *********")
        # data preprocessing
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "lchen001/AIME1983_2024",
                    "split": "train",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(
                                name_mapping={
                                    "Question": "prompt",
                                    "Answer": "ground_truth",
                                }
                            ),
                            MultiplyTransform(n_repeats=2),
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
            max_concurrent=max_concurrent,
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
            
                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": [
                            "NumericMatch_result",
                        ],
                        "first_groupby": "data_repeat_id",
                        "filename_base": "NumericMatch",
                        "normalize": True,
                    },
                ),
                
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": [
                            "NumericMatch_result",
                        ],
                        "group_by": "Year",
                        "filename_base": "NumericMatch_GroupBy",
                    },
                ),

                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": [
                            "NumericMatch_result",
                        ],
                        "first_groupby": "ID",
                        "second_groupby": "Part",
                        "filename_base": "NumericMatch_GroupBy_Part",
                        "normalize": True,
                    },
                ),
                
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "usage_completion"
                        ],
                        "first_groupby": "ID",
                        "filename_base": "UsageCompletion",
                         "agg_fn": "mean"
                    },
                ),

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
                    "path": os.path.join(self.data_post_processing.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [                         
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
        # Second, compute eaxct match
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
                        "first_groupby": "data_repeat_id",
                        "filename_base": "MajorityVote",
                        "normalize": True,
                    },
                ),
                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": [
                            "NumericMatch_result",
                        ],
                        "first_groupby": "data_repeat_id",
                        "second_groupby": "Year",
                        "filename_base": "MajorityVote_byyear",
                        "normalize": True,
                    },
                ),
                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": [
                            "NumericMatch_result",
                        ],
                        "first_groupby": "data_repeat_id",
                        "second_groupby": "Part",
                        "filename_base": "MajorityVote_bypart",
                        "normalize": True,
                    },
                ),

                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "usage_completion"
                        ],
                        "first_groupby": "ID",
                        "filename_base": "UsageCompletion",
                         "agg_fn": "mean"
                    },
                ),              
            ],
            output_dir=os.path.join(self.log_dir, "eval_report_majorityVote"),
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
                # the first three reports aggregate results by ID and take the best out of N
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "NumericMatch_result_numeric"
                        ],
                        "first_groupby": "ID",
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
                        "first_groupby": "ID", 
                        "second_groupby": "Year",
                        "filename_base": "NumericMatch_BestOfN_GroupBy_Year",
                        "agg_fn": "max"
                    },
                ),

                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": [
                            "NumericMatch_result",
                        ],
                        "first_groupby": "ID",
                        "second_groupby": "Part",
                        "filename_base": "NumericMatch_BestOfN_GroupBy_Part",
                        "normalize": True,
                    },
                ),
                
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "usage_completion"
                        ],
                        "first_groupby": "ID",
                        "filename_base": "UsageCompletion_BestOfN",
                         "agg_fn": "mean"
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
                # the first three reports aggregate results by ID and take the best out of N
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "NumericMatch_result_numeric"
                        ],
                        "first_groupby": "ID",
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
                        "first_groupby": "ID", 
                        "second_groupby": "Year",
                        "filename_base": "NumericMatch_WorstOfN_GroupBy_Year",
                        "agg_fn": "min"
                    },
                ),

                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": [
                            "NumericMatch_result",
                        ],
                        "first_groupby": "ID",
                        "second_groupby": "Part",
                        "filename_base": "NumericMatch_WorstOfN_GroupBy_Part",
                        "normalize": True,
                    },
                ),

                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "usage_completion"
                        ],
                        "first_groupby": "ID",
                        "filename_base": "UsageCompletion_WorstOfN",
                         "agg_fn": "mean"
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
                self.mv_evalreporting_comp,
                self.posteval_data_post_processing_comp,
                self.bon_evalreporting_comp,
                self.won_evalreporting_comp
            ],
            self.log_dir,
        )

class AIME_PIPLELINE_HYBRIDEXTRACTION(AIME_PIPELINE):
    """This class specifies the config for running AIME with a hybrid answer extraction"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)

        self.preeval_data_post_processing_comp = DataProcessingConfig(
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
                                    "model_output": "raw_model_output",
                                }
                            ),
                            AddColumn("model_output"),
                            AIMEExtractAnswer("raw_model_output", "model_output"),
                            ExtractUsageTransform(model_config),  
                            ImputeNA(columns="model_output", value="")
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "preeval_data_post_processing_output"),
        )

        self.filter_empty_answer = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.preeval_data_post_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={"prompt": "initial_prompt"}),
                            AddColumn(column_name="prompt")
                        ]
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__),
                "../prompt_templates/aime_templates/extract_aime_answer.jinja",
            ),
            output_dir=os.path.join(self.log_dir, "filter_empty_answer"),
        )       
        
        self.inference_llm_answer_extract = InferenceConfig(
            component_type=Inference,
            model_config=OAI_GPT4O_2024_11_20_CONFIG,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {"path": os.path.join(self.filter_empty_answer.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(self.log_dir, "llm_answer_extract_inference_result"),
            max_concurrent=5
            )
        
        self.data_join = DataJoinConfig(
            component_type=DataJoin,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.preeval_data_post_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            other_data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_llm_answer_extract.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            # drop all columns except the uid and model_output
                            RunPythonTransform("df = df[[col for col in ['data_repeat_id','data_point_id', 'model_output'] if col in df.columns]]"),
                            
                            ColumnRename(
                                name_mapping={
                                    "model_output": "raw_output",
                                }
                            ),

                            AIMEExtractAnswer("raw_output", "model_output"),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_join_output"),
            pandas_merge_args={"on": ['data_repeat_id', 'data_point_id'], "how": "left"},
            )

        # post process the response to extract the answer
        self.data_post_processing = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.data_join.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                           RunPythonTransform("df['model_output'] = df.apply(lambda row: row['model_output'] if 'model_output_x' not in row else row['model_output_y'] if row['model_output_x'] == '' else row['model_output_x'], axis=1)"),                       
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_post_processing_output"),
        )

        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.preeval_data_post_processing_comp,
                self.filter_empty_answer,
                self.inference_llm_answer_extract,
                self.data_join,
                self.data_post_processing,
                self.evalreporting_comp,
                self.data_post_processing_addmv,
                self.mv_evalreporting_comp,
                self.posteval_data_post_processing_comp,
                self.bon_evalreporting_comp,
                self.won_evalreporting_comp
            ],
            self.log_dir,
        )

class AIME_PIPLELINE_HYBRIDEXTRACTION5Run_2025(AIME_PIPLELINE_HYBRIDEXTRACTION):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "lchen001/AIME2025",
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

        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=5)
        )
        
        # Configure the pipeline; this is necessary for resume_from to work
        return pipeline

class AIME_PIPLELINE_HYBRIDEXTRACTION5Run(AIME_PIPLELINE_HYBRIDEXTRACTION):
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
        # data preprocessing
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "lchen001/AIME2025",
                    "split": "train",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(
                                name_mapping={
                                    "Question": "prompt",
                                    "Answer": "ground_truth",
                                }
                            ),
                            #SamplerTransform( random_seed=0,sample_count=2),
                        ],
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/aime_templates/Template_1clean.jinja"
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
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
        
class AIME_PIPELINE5Run_2025_Direct(AIME_PIPELINE):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "lchen001/AIME2025",
                    "split": "train",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(
                                name_mapping={
                                    "Question": "prompt",
                                    "Answer": "ground_truth",
                                }
                            ),
                            #SamplerTransform( random_seed=0,sample_count=2),
                        ],
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/aime_templates/Template_1direct.jinja"
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
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

class AIME_PIPELINE50Run_2025(AIME_PIPELINE5Run_2025):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms[-1] = MultiplyTransform(n_repeats=50)
        return pipeline

class AIME_PIPELINE16Run_2025(AIME_PIPELINE5Run_2025):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms[-1] = MultiplyTransform(n_repeats=16)
        return pipeline

class AIME_PIPELINE32Run_2025(AIME_PIPELINE5Run_2025):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms[-1] = MultiplyTransform(n_repeats=32)
        return pipeline
        
class AIME_PIPELINE64Run_2025(AIME_PIPELINE5Run_2025):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms[-1] = MultiplyTransform(n_repeats=64)
        return pipeline
        
class AIME_PIPELINE128Run_2025(AIME_PIPELINE5Run_2025):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms[-1] = MultiplyTransform(n_repeats=128)
        return pipeline

class AIME_PIPELINE256Run_2025(AIME_PIPELINE5Run_2025):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms[-1] = MultiplyTransform(n_repeats=256)
        return pipeline

class AIME_PIPELINE512Run_2025(AIME_PIPELINE5Run_2025):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms[-1] = MultiplyTransform(n_repeats=512)
        return pipeline

class AIME_PIPELINE1024Run_2025(AIME_PIPELINE5Run_2025):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms[-1] = MultiplyTransform(n_repeats=1024)
        return pipeline
        
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
