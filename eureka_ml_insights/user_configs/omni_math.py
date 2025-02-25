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
    PipelineConfig,
    PromptProcessingConfig,
    ModelConfig,
)
from ..configs.experiment_config import ExperimentConfig

class Omni_Math_PIPELINE(ExperimentConfig):
    """This class specifies the config for running any benchmark on any model"""

    def configure_pipeline(self, model_config=None, resume_from=None, eval_resume_from=None, eval_model_config=None, **kwargs) -> PipelineConfig:
        # data preprocessing
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            prompt_template_path=os.path.join(os.path.dirname(__file__), "../prompt_templates/omni_math_templates/omni_math_cot.jinja"),
            #prompt_template_path=os.path.join(os.path.dirname(__file__), "../prompt_templates/omni_math_templates/omni_math_brief.jinja"),
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                   "path": "KbsdJames/Omni-MATH",
                   "split": "test",
                   "transform": SequenceTransform([
                    # SamplerTransform(sample_count=100, random_seed=99),
                    MultiplyTransform(n_repeats=1),
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
                    "path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl")
                },
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
            max_concurrent=5,
        )

        # eval data preprocessing
        self.eval_data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            prompt_template_path=os.path.join(os.path.dirname(__file__), "../prompt_templates/omni_math_templates/omni_math_gpt_eval.jinja"),
            data_reader_config=DataSetConfig(
                DataReader,
                {"path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                 "transform": SequenceTransform([
                    CopyColumn("model_output", "generated_solution"),
                    ColumnRename(name_mapping={"n_output_tokens":"gen_solution_n_output_tokens",
                                               "usage": "gen_solution_usage"}),
                    # SamplerTransform(sample_count=5, random_seed=99),
                 ])},
            ),
            output_dir=os.path.join(self.log_dir, "eval_data_processing_output"),
        )

        # inference component
        self.eval_inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=eval_model_config,
            # model_config=TRAPI_GPT4O_2024_11_20_EVAL_CONFIG,
            # model_config=OAI_GPT4O_2024_11_20_CONFIG,
            #model_config=TRAPI_GCR_SHARED_O1_CONFIG,
            data_loader_config=DataSetConfig(
                DataLoader,
                {"path": os.path.join(self.eval_data_processing_comp.output_dir, "transformed_data.jsonl")
                },
            ),
            output_dir=os.path.join(self.log_dir, "eval_inference_result"),
            resume_from=eval_resume_from,
            max_concurrent=5,
        )
        
        self.eval_inf_data_processing_comp = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.eval_inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            ExtractUsageTransform(model_config, prepend_completion_read_col="gen_solution_"),
                            ColumnRename(
                                name_mapping={
                                    "model_output": "raw_output",
                                }
                            ),
                            AddColumn("model_output"),
                            AddColumn("model_solution"),
                            Omni_Math_ParseLabel("raw_output", "OmniMath_correctness"),
                            Omni_Math_ParseSolution("raw_output", "model_output"),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "eval_inf_data_processing_output"),
        )

        # Configure the evaluation and reporting component for evaluation and dataset level aggregation
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.eval_inf_data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            aggregator_configs=[
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": [
                            "OmniMath_correctness",
                        ],
                        "filename_base": "Correctness_SeparateRuns",
                        "group_by": "data_repeat_id",
                    },
                ),
                # the next three reports take the average and std for all repeats
                # the resulting numbers are the average and std of N pass@1 scores, where N is number of repeats
                AggregatorConfig(BiLevelAggregator, 
                    {
                        "column_names": [
                            "OmniMath_correctness"
                        ], 
                        "first_groupby": "data_repeat_id", 
                        "filename_base": "Correctness_Avg",
                        "agg_fn": "mean"
                    }),
                AggregatorConfig(BiLevelAggregator, 
                    {
                        "column_names": [
                            "OmniMath_correctness"
                        ], 
                        "first_groupby": ["data_repeat_id", "difficulty"], 
                        "second_groupby": "difficulty",
                        "filename_base": "Correctness_Avg_by_difficulty",
                        "agg_fn": "mean"
                    }), 
                AggregatorConfig(BiLevelAggregator, 
                    {
                        "column_names": [
                            "OmniMath_correctness"
                        ], 
                        "first_groupby": ["data_repeat_id", "source"], 
                        "second_groupby": "source",
                        "filename_base": "Correctness_Avg_by_source",
                        "agg_fn": "mean"
                    }),                
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
                        "first_groupby": ["data_point_id", "difficulty"],
                        "second_groupby": "difficulty",
                        "filename_base": "UsageCompletion_by_difficulty",
                        "agg_fn": "mean"
                    }),
                AggregatorConfig(BiLevelAggregator, 
                    {
                        "column_names": ["usage_completion"], 
                        "first_groupby": ["data_point_id", "source"],
                        "second_groupby": "source",
                        "filename_base": "UsageCompletion_by_difficulty_source",
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
                    "path": os.path.join(self.eval_inf_data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            aggregator_configs=[
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "OmniMath_correctness",
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
                            "OmniMath_correctness",
                        ],
                        "first_groupby": "data_point_id",
                        "second_groupby": "difficulty",
                        "filename_base": "Correctness_BestOfN_by_difficulty",
                        "normalize": True,
                        "agg_fn": "max",
                    },
                ),
                
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "OmniMath_correctness",
                        ],
                        "first_groupby": "data_point_id",
                        "second_groupby": "source",
                        "filename_base": "Correctness_BestOfN_by_source",
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
                    "path": os.path.join(self.eval_inf_data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            aggregator_configs=[
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "OmniMath_correctness",
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
                            "OmniMath_correctness",
                        ],
                        "first_groupby": "data_point_id",
                        "second_groupby": "difficulty",
                        "filename_base": "Correctness_WorstOfN_by_difficulty",
                        "normalize": True,
                        "agg_fn": "min",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "OmniMath_correctness",
                        ],
                        "first_groupby": "data_point_id",
                        "second_groupby": "source",
                        "filename_base": "Correctness_WorstOfN_by_source",
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
                    "path": os.path.join(self.eval_inf_data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(
                                name_mapping={
                                    "model_output": "model_output_onerun",
                                }
                            ),
                            AddColumn("model_output"),
                            MajorityVoteTransform(model_output_col="model_output_onerun", model_label_column="OmniMath_correctness"),
                            CopyColumn("majority_vote", "model_output"),
                            CopyColumn("majority_label", "OmniMath_correctness_majority_vote"),
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
                            "OmniMath_correctness_majority_vote",
                        ],
                        "filename_base": "Correctness_MajVote",
                    },
                ),
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": [
                            "OmniMath_correctness_majority_vote",
                        ],
                        "filename_base": "Correctness_MajVote_by_difficulty",
                        "group_by": "difficulty",
                    },
                ),
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": [
                            "OmniMath_correctness_majority_vote",
                        ],
                        "filename_base": "Correctness_MajVote_by_source",
                        "group_by": "source",
                    },
                ),
                AggregatorConfig(CountAggregator, 
                    {
                        "column_names": [
                            "count",
                        ], 
                        "group_by": "difficulty", 
                        "filename_base": "NumExamples_by_difficulty",
                    }),
                AggregatorConfig(CountAggregator, 
                    {
                        "column_names": [
                            "count",
                        ], 
                        "group_by": "source", 
                        "filename_base": "NumExamples_by_source",
                    }),
            ],
            output_dir=os.path.join(self.log_dir, "majvote_eval_report"),
        )

        self.domain_eval_data_processing_comp = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.maj_vote_data_post_processing.output_dir, "transformed_data.jsonl"),
                    # "path": resume_from,
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            RunPythonTransform("df['highlevel_domain'] = df['domain'].apply(lambda x: list(set([y.split('->')[0] for y in x])))"),
                            RunPythonTransform("df['sub_domain'] = df['domain'].apply(lambda x: list(set([y.split('->')[1] for y in x])))"),
                            #RunPythonTransform("df['sec_sub_domain'] = df['domain'].apply(lambda x: list(set([y.split('->')[2] for y in x])))"),
                            # RunPythonTransform("df = df.explode(['highlevel_domain', 'sub_domain', 'sec_sub_domain'])"),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "domain_eval_data_processing_output"),
        )

        
        # Configure the evaluation and reporting component for domain level aggregation
        self.domain_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.domain_eval_data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            RunPythonTransform("df = df.explode(['highlevel_domain'])"), #, 'sub_domain', 'sec_sub_domain'])"),
                            # RunPythonTransform("df = df[df['data_repeat_id'] == 'repeat_0']"),
                        ]
                    ),
                },
            ),
            aggregator_configs=[
                AggregatorConfig(BiLevelAggregator, 
                    {
                        "column_names": [
                            "OmniMath_correctness"
                        ], 
                        "first_groupby": ["data_repeat_id", "highlevel_domain"], 
                        "second_groupby": "highlevel_domain",
                        "filename_base": "Correctness_Avg_by_domain",
                        "agg_fn": "mean"
                    }), 
                AggregatorConfig(BiLevelAggregator, 
                    {
                        "column_names": ["usage_completion"], 
                        "first_groupby": ["data_point_id", "highlevel_domain"],
                        "second_groupby": "highlevel_domain",
                        "filename_base": "UsageCompletion_by_highlevel_domain",
                        "agg_fn": "mean"
                    }),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "OmniMath_correctness",
                        ],
                        "first_groupby": "data_point_id",
                        "second_groupby": "highlevel_domain",
                        "filename_base": "Correctness_BestOfN_by_highlevel_domain",
                        "normalize": True,
                        "agg_fn": "max",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "OmniMath_correctness",
                        ],
                        "first_groupby": "data_point_id",
                        "second_groupby": "highlevel_domain",
                        "filename_base": "Correctness_WorstOfN_by_highlevel_domain",
                        "normalize": True,
                        "agg_fn": "min",
                    },
                ),
                
            ],
            output_dir=os.path.join(self.log_dir, "eval_report_by_domain"),
        )

        self.domain_majvote_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.domain_eval_data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            RunPythonTransform("df = df.explode(['highlevel_domain'])"), #, 'sub_domain', 'sec_sub_domain'])"),
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
                            "OmniMath_correctness_majority_vote",
                        ],
                        "filename_base": "Correctness_MajVote_by_highlevel_domain",
                        "group_by": "highlevel_domain",
                    },
                ),
                AggregatorConfig(CountAggregator, 
                    {
                        "column_names": [
                            "count",
                        ], 
                        "group_by": "highlevel_domain", 
                        "filename_base": "NumExamples_by_highlevel_domain",
                    }
                ),
            ],
            output_dir=os.path.join(self.log_dir, "majvote_eval_report_by_domain"),
        )

        # Configure the evaluation and reporting component for domain level aggregation
        self.sub_domain_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.domain_eval_data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            RunPythonTransform("df = df.explode(['sub_domain'])"), #, 'sub_domain', 'sec_sub_domain'])"),
                            # RunPythonTransform("df = df[df['data_repeat_id'] == 'repeat_0']"),
                        ]
                    ),
                },
            ),
            aggregator_configs=[
                AggregatorConfig(BiLevelAggregator, 
                    {
                        "column_names": [
                            "OmniMath_correctness"
                        ], 
                        "first_groupby": ["data_repeat_id", "sub_domain"], 
                        "second_groupby": "sub_domain",
                        "filename_base": "Correctness_Avg_by_sub_domain",
                        "agg_fn": "mean"
                    }),
                AggregatorConfig(BiLevelAggregator, 
                    {
                        "column_names": ["usage_completion"], 
                        "first_groupby": ["data_point_id", "sub_domain"],
                        "second_groupby": "sub_domain",
                        "filename_base": "UsageCompletion_by_sub_domain",
                        "agg_fn": "mean"
                    }),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "OmniMath_correctness",
                        ],
                        "first_groupby": "data_point_id",
                        "second_groupby": "sub_domain",
                        "filename_base": "Correctness_BestOfN_by_sub_domain",
                        "normalize": True,
                        "agg_fn": "max",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "OmniMath_correctness",
                        ],
                        "first_groupby": "data_point_id",
                        "second_groupby": "sub_domain",
                        "filename_base": "Correctness_WorstOfN_by_sub_domain",
                        "normalize": True,
                        "agg_fn": "min",
                    },
                ),                
            ],
            output_dir=os.path.join(self.log_dir, "eval_report_by_sub_domain"),
        )

        self.sub_domain_majvote_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.domain_eval_data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            RunPythonTransform("df = df.explode(['sub_domain'])"), #, 'sub_domain', 'sec_sub_domain'])"),
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
                            "OmniMath_correctness_majority_vote",
                        ],
                        "filename_base": "Correctness_MajVote_by_sub_domain",
                        "group_by": "sub_domain",
                    },
                ),
                AggregatorConfig(CountAggregator, 
                    {
                        "column_names": [
                            "count",
                        ], 
                        "group_by": "sub_domain", 
                        "filename_base": "NumExamples_by_sub_domain",
                    }
                ),
            ],
            output_dir=os.path.join(self.log_dir, "majvote_eval_report_by_sub_domain"),
        )

        # Configure the evaluation and reporting component for domain level aggregation
        self.sec_sub_domain_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.domain_eval_data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            RunPythonTransform("df = df.explode(['sec_sub_domain'])"), #, 'sub_domain', 'sec_sub_domain'])"),
                            # RunPythonTransform("df = df[df['data_repeat_id'] == 'repeat_0']"),
                        ]
                    ),
                },
            ),
            aggregator_configs=[
                AggregatorConfig(BiLevelAggregator,
                    {
                        "column_names": [
                            "OmniMath_correctness"
                        ],
                        "first_groupby": ["data_repeat_id", "sec_sub_domain"],
                        "second_groupby": "sec_sub_domain",
                        "filename_base": "Correctness_Avg_by_sec_sub_domain",
                        "agg_fn": "mean"
                    }),
                AggregatorConfig(BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": ["data_point_id", "sec_sub_domain"],
                        "second_groupby": "sec_sub_domain",
                        "filename_base": "UsageCompletion_by_sec_sub_domain",
                        "agg_fn": "mean"
                    }),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "OmniMath_correctness",
                        ],
                        "first_groupby": "data_point_id",
                        "second_groupby": "sec_sub_domain",
                        "filename_base": "Correctness_BestOfN_by_sec_sub_domain",
                        "normalize": True,
                        "agg_fn": "max",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "OmniMath_correctness",
                        ],
                        "first_groupby": "data_point_id",
                        "second_groupby": "sec_sub_domain",
                        "filename_base": "Correctness_WorstOfN_by_sec_sub_domain",
                        "normalize": True,
                        "agg_fn": "min",
                    },
                )
                
            ],
            output_dir=os.path.join(self.log_dir, "eval_report_by_sec_sub_domain"),
        )

        self.sec_sub_domain_majvote_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.domain_eval_data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            RunPythonTransform("df = df.explode(['sec_sub_domain'])"), #, 'sub_domain', 'sec_sub_domain'])"),
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
                            "OmniMath_correctness_majority_vote",
                        ],
                        "filename_base": "Correctness_MajVote_by_sec_sub_domain",
                        "group_by": "sec_sub_domain",
                    },
                ),
                AggregatorConfig(CountAggregator,
                    {
                        "column_names": [
                            "count",
                        ],
                        "group_by": "sec_sub_domain",
                        "filename_base": "NumExamples_by_sec_sub_domain",
                    }
                ),
            ],
            output_dir=os.path.join(self.log_dir, "majvote_eval_report_by_sec_sub_domain"),
        )

        # Configure the pipeline
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.eval_data_processing_comp,
                self.eval_inference_comp,
                self.eval_inf_data_processing_comp,
                self.evalreporting_comp,
                self.bon_evalreporting_comp,
                self.won_evalreporting_comp,
                self.maj_vote_data_post_processing,
                self.majvote_evalreporting_comp,
                self.domain_eval_data_processing_comp,
                self.domain_evalreporting_comp,
                self.domain_majvote_evalreporting_comp,
                self.sub_domain_evalreporting_comp,
                self.sub_domain_majvote_evalreporting_comp,
                #self.sec_sub_domain_evalreporting_comp,
                #self.sec_sub_domain_majvote_evalreporting_comp,
            ],
            self.log_dir,
        )

class Omni_Math_Parallel_PIPELINE(Omni_Math_PIPELINE):
    """This class specifies the config for running Omni Math benchmark 5 repeated times"""

    def configure_pipeline(
            self, model_config: ModelConfig, resume_from: str = None, eval_resume_from: str = None, eval_model_config: ModelConfig = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from, eval_resume_from=eval_resume_from, eval_model_config=eval_model_config)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms[-1] = MultiplyTransform(n_repeats=5)
        return pipeline


class Omni_Math_ExtractUsage_PIPELINE(Omni_Math_PIPELINE):
    """This class specifies the config for running Omni Math benchmark 5 repeated times"""

    def configure_pipeline(
            self, model_config: ModelConfig, resume_from: str = None, eval_model_config: ModelConfig = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        # pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from, eval_model_config=eval_model_config)
        # data preprocessing
        self.usage_data_processing_comp = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": resume_from,
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            ExtractUsageTransform(model_config),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "usage_data_processing_output"),
        )

        # Configure the evaluation and reporting component for evaluation and dataset level aggregation
        self.usage_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.usage_data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            aggregator_configs=[
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
                        "first_groupby": ["data_point_id", "difficulty"],
                        "second_groupby": "difficulty",
                        "filename_base": "UsageCompletion_by_difficulty",
                        "agg_fn": "mean"
                    }),
                AggregatorConfig(BiLevelAggregator, 
                    {
                        "column_names": ["usage_completion"], 
                        "first_groupby": ["data_point_id", "source"],
                        "second_groupby": "source",
                        "filename_base": "UsageCompletion_by_difficulty_source",
                        "agg_fn": "mean"
                    }),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # Aggregate the results by best of n
        self.usage_bon_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.usage_data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            aggregator_configs=[
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

        self.usage_domain_eval_data_processing_comp = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.usage_data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            RunPythonTransform("df['highlevel_domain'] = df['domain'].apply(lambda x: list(set([y.split('->')[0] for y in x])))"),
                            RunPythonTransform("df['sub_domain'] = df['domain'].apply(lambda x: list(set([y.split('->')[1] for y in x])))"),
                            #RunPythonTransform("df['sec_sub_domain'] = df['domain'].apply(lambda x: list(set([y.split('->')[2] for y in x])))"),
                            # RunPythonTransform("df = df.explode(['highlevel_domain', 'sub_domain', 'sec_sub_domain'])"),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "domain_eval_data_processing_output"),
        )

        
        # Configure the evaluation and reporting component for domain level aggregation
        self.usage_domain_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.usage_data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            RunPythonTransform("df = df.explode(['highlevel_domain'])"), #, 'sub_domain', 'sec_sub_domain'])"),
                            # RunPythonTransform("df = df[df['data_repeat_id'] == 'repeat_0']"),
                        ]
                    ),
                },
            ),
            aggregator_configs=[
                AggregatorConfig(BiLevelAggregator, 
                    {
                        "column_names": ["usage_completion"], 
                        "first_groupby": ["data_point_id", "highlevel_domain"],
                        "second_groupby": "highlevel_domain",
                        "filename_base": "UsageCompletion_by_highlevel_domain",
                        "agg_fn": "mean"
                    }),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report_by_domain"),
        )
        # Configure the evaluation and reporting component for domain level aggregation
        self.usage_sub_domain_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.usage_data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            RunPythonTransform("df = df.explode(['sub_domain'])"), #, 'sub_domain', 'sec_sub_domain'])"),
                            # RunPythonTransform("df = df[df['data_repeat_id'] == 'repeat_0']"),
                        ]
                    ),
                },
            ),
            aggregator_configs=[
                AggregatorConfig(BiLevelAggregator, 
                    {
                        "column_names": ["usage_completion"], 
                        "first_groupby": ["data_point_id", "sub_domain"],
                        "second_groupby": "sub_domain",
                        "filename_base": "UsageCompletion_by_sub_domain",
                        "agg_fn": "mean"
                    }),                
            ],
            output_dir=os.path.join(self.log_dir, "eval_report_by_sub_domain"),
        )
        # Configure the evaluation and reporting component for domain level aggregation
        self.usage_sec_sub_domain_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.usage_data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            RunPythonTransform("df = df.explode(['sec_sub_domain'])"), #, 'sub_domain', 'sec_sub_domain'])"),
                            # RunPythonTransform("df = df[df['data_repeat_id'] == 'repeat_0']"),
                        ]
                    ),
                },
            ),
            aggregator_configs=[
                AggregatorConfig(BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": ["data_point_id", "sec_sub_domain"],
                        "second_groupby": "sec_sub_domain",
                        "filename_base": "UsageCompletion_by_sec_sub_domain",
                        "agg_fn": "mean"
                    }),                
            ],
            output_dir=os.path.join(self.log_dir, "eval_report_by_sec_sub_domain"),
        )

        # Configure the pipeline
        return PipelineConfig(
            [
                self.usage_data_processing_comp,
                self.usage_evalreporting_comp,
                self.usage_bon_evalreporting_comp,
                self.usage_domain_eval_data_processing_comp,
                self.usage_domain_evalreporting_comp,
                self.usage_sub_domain_evalreporting_comp,
            ],
            self.log_dir,
        )
