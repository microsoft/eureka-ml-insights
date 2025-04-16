import os

from eureka_ml_insights.configs.experiment_config import ExperimentConfig
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing, DataProcessing, DataJoin

from eureka_ml_insights.data_utils import (
    AddColumn,
    HFDataReader,  
    DataLoader,  
    MMDataLoader,
    CopyColumn,
    ExtractUsageTransform,
    ReplaceStringsTransform,
    RunPythonTransform,
    MajorityVoteTransform,
    ColumnRename,
    DataReader,
    ExtractQuestionOptions,
    ExtractAnswerSpatialMapAndMaze,
    MultiplyTransform,
    SequenceTransform,
    RegexTransform,
)
from eureka_ml_insights.metrics import SubstringExistsMatch, BiLevelAggregator, BiLevelCountAggregator, CountAggregator

from eureka_ml_insights.configs import (
    AggregatorConfig,
    DataProcessingConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
    DataJoinConfig,    
)
from eureka_ml_insights.configs.model_configs import OAI_GPT4O_2024_11_20_CONFIG

"""This file contains example user defined configuration classes for the spatial map task.
In order to define a new configuration, a new class must be created that directly or indirectly
 inherits from UserDefinedConfig and the user_init method should be implemented.
You can inherit from one of the existing user defined classes below and override the necessary
attributes to reduce the amount of code you need to write.

The user defined configuration classes are used to define your desired *pipeline* that can include
any number of *component*s. Find *component* options in the core module.

Pass the name of the class to the main.py script to run the pipeline.
"""


class SPATIAL_MAP_PIPELINE(ExperimentConfig):
    """This method is used to define an eval pipeline with inference and metric report components,
    on the spatial map dataset."""

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None) -> PipelineConfig:
        # Configure the data processing component.
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "microsoft/VISION_LANGUAGE",
                    "split": "val_noinstruction",
                    "tasks": "spatial_map",
                    "transform": MultiplyTransform(n_repeats=5),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__),
                "../../prompt_templates/vision_language_templates/basic.jinja",
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        # Configure the inference component
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {
                    "path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl"),
                },
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
            max_concurrent=10,
        )

        self.preeval_data_post_processing_comp = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            ExtractUsageTransform(model_config),                        
                            ExtractQuestionOptions(
                                    prompt_column_name="prompt",
                                    extracted_options_column_name="target_options_answers",
                            ),
                            ColumnRename(name_mapping={"model_output": "model_output_raw"}),
                            ExtractAnswerSpatialMapAndMaze(
                                answer_column_name="model_output_raw",
                                extracted_answer_column_name="model_output",
                                extracted_options_column_name="target_options_answers",
                            ),
                        ],
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
                            RunPythonTransform("df = df[df['model_output'] == '']"),
                            ColumnRename(name_mapping={"prompt": "initial_prompt"}),
                            AddColumn(column_name="prompt")
                        ]
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__),
                "../../prompt_templates/vision_language_templates/extract_answer.jinja",
            ),
            output_dir=os.path.join(self.log_dir, "filter_empty_answer"),
        )

        self.inference_llm_answer_extract = InferenceConfig(
            component_type=Inference,
            model_config=OAI_GPT4O_2024_11_20_CONFIG,
            data_loader_config=DataSetConfig(
                DataLoader,
                {"path": os.path.join(self.filter_empty_answer.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(self.log_dir, "llm_answer_extract_inference_result"),
            max_concurrent=1
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
                            RegexTransform(
                                columns="model_output",
                                prompt_pattern=r"Final Answer:\s*(.+)",
                                ignore_case=True,
                            ),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_join_output"),
            pandas_merge_args={"on": ['data_repeat_id', 'data_point_id'], "how": "left"},
        )        

        # Configure the evaluation and reporting component.
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.data_join.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            # consolidate model_output_y to replace the original model_output whenever empty
                            # the initial if statement checks whether there has been a join beforehand
                            RunPythonTransform("df['model_output'] = df.apply(lambda row: row['model_output'] if 'model_output_x' not in row else row['model_output_y'] if row['model_output_x'] == '' else row['model_output_x'], axis=1)"),
                        ]
                    ),
                },
            ),
            metric_config=MetricConfig(SubstringExistsMatch),
            aggregator_configs=[
                # the first three reports aggregate the metrics per experiment repeat
                # each repeat can be considered as an individual pass@1 score
                AggregatorConfig(CountAggregator, 
                    {
                        "column_names": ["SubstringExistsMatch_result"], 
                        "group_by": "data_repeat_id", 
                        "filename_base": "SubstringExistsMatch_SeparateRuns",
                        "normalize": True
                    }),
                AggregatorConfig(CountAggregator, 
                    {
                        "column_names": ["SubstringExistsMatch_result"], 
                        "group_by": ["data_repeat_id", "task"], 
                        "filename_base": "SubstringExistsMatch_GroupBy_task_SeparateRuns", 
                        "normalize": True
                    }),
                # the next three reports take the average and std for all repeats
                # the resulting numbers are the average and std of N pass@1 scores, where N is number of repeats
                AggregatorConfig(BiLevelCountAggregator, 
                    {
                        "column_names": ["SubstringExistsMatch_result"], 
                        "first_groupby": "data_repeat_id", 
                        "filename_base": "SubstringExistsMatch_AllRuns",
                        "normalize": True
                    }),
                AggregatorConfig(BiLevelCountAggregator, 
                    {
                        "column_names": ["SubstringExistsMatch_result"], 
                        "first_groupby": ["data_repeat_id",    "task"], 
                        "second_groupby": "task",
                        "filename_base": "SubstringExistsMatch_GroupBy_task_AllRuns", 
                        "normalize": True
                    }),
                # three similar reports for average completion usage
                AggregatorConfig(BiLevelAggregator, 
                    {
                        "column_names": ["usage_completion"], 
                        "first_groupby": "data_repeat_id", 
                        "filename_base": "UsageCompletion_AllRuns",
                        "agg_fn": "mean"
                    }),
                AggregatorConfig(BiLevelAggregator, 
                    {
                        "column_names": ["usage_completion"], 
                        "first_groupby": ["data_repeat_id", "task"], 
                        "second_groupby": "task",
                        "filename_base": "UsageCompletion_GroupBy_task_AllRuns", 
                        "agg_fn": "mean"
                    }),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
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
                                column_name_src="SubstringExistsMatch_result",
                                column_name_dst="SubstringExistsMatch_result_numeric",
                            ),
                        ReplaceStringsTransform(
                                columns=["SubstringExistsMatch_result_numeric"],
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
                            "SubstringExistsMatch_result_numeric"
                        ],
                        "first_groupby": "data_point_id",
                        "filename_base": "SubstringExistsMatch_BestOfN",
                        "agg_fn": "max"
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "SubstringExistsMatch_result_numeric"
                        ],
                        "first_groupby": "data_point_id", 
                        "second_groupby": "task",
                        "filename_base": "SubstringExistsMatch_BestOfN_GroupBy_task",
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
                            "SubstringExistsMatch_result_numeric"
                        ],
                        "first_groupby": "data_point_id",
                        "filename_base": "SubstringExistsMatch_WorstOfN",
                        "agg_fn": "min"
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "SubstringExistsMatch_result_numeric"
                        ],
                        "first_groupby": "data_point_id", 
                        "second_groupby": "task",
                        "filename_base": "SubstringExistsMatch_WorstOfN_GroupBy_task",
                        "agg_fn": "min"
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "worstofn_eval_report"),
        )

        # aggregate the output by majority vote
        self.data_post_processing_mv = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.evalreporting_comp.output_dir, "metric_results.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            MajorityVoteTransform(model_output_col="model_output"),
                            ColumnRename(
                                name_mapping={
                                    "model_output": "model_output_onerun",
                                    "majority_vote": "model_output",
                                }
                            ),
                            RunPythonTransform("df = df[df['data_repeat_id'] == 'repeat_0']")
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_post_processing_mv"),
        )

        # Configure the evaluation and reporting component for majority vote.
        self.mv_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.data_post_processing_mv.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(SubstringExistsMatch),
            aggregator_configs=[
                # these three reports aggregate the metrics for the majority vote results
                AggregatorConfig(CountAggregator, 
                    {
                        "column_names": ["SubstringExistsMatch_result"], 
                        "filename_base": "MajorityVote",
                        "normalize": True
                    }),
                AggregatorConfig(CountAggregator, 
                    {
                        "column_names": ["SubstringExistsMatch_result"], 
                        "group_by": ["task"], 
                        "filename_base": "MajorityVote_GroupBy_task", 
                        "normalize": True
                    }),
            ],
            output_dir=os.path.join(self.log_dir, "majorityvote_eval_report"),
        )

        # Configure the pipeline
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.preeval_data_post_processing_comp,
                self.filter_empty_answer,
                self.inference_llm_answer_extract,
                self.data_join,                
                self.evalreporting_comp,
                self.posteval_data_post_processing_comp,
                self.bon_evalreporting_comp,
                self.won_evalreporting_comp,
                self.data_post_processing_mv,
                self.mv_evalreporting_comp
            ],
            self.log_dir,
        )
class SPATIAL_MAP_COT_PIPELINE(SPATIAL_MAP_PIPELINE):
    """This class extends SPATIAL_MAP_PIPELINE to use a COT prompt."""

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None) -> PipelineConfig:
        config = super().configure_pipeline(model_config, resume_from)
        self.data_processing_comp.prompt_template_path=os.path.join(
                os.path.dirname(__file__),
                "../../prompt_templates/vision_language_templates/cot.jinja",
            )
        return config

class SPATIAL_MAP_TEXTONLY_PIPELINE(SPATIAL_MAP_PIPELINE):
    """This class extends SPATIAL_MAP_PIPELINE to use text only data."""

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None) -> PipelineConfig:
        config = super().configure_pipeline(model_config, resume_from)
        self.data_processing_comp.data_reader_config.init_args["tasks"] = (
            "spatial_map_text_only"
        )
        return config

class SPATIAL_MAP_COT_TEXTONLY_PIPELINE(SPATIAL_MAP_COT_PIPELINE):
    """This class extends SPATIAL_MAP_PIPELINE to use text only data."""

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None) -> PipelineConfig:
        config = super().configure_pipeline(model_config, resume_from)
        self.data_processing_comp.data_reader_config.init_args["tasks"] = (
            "spatial_map_text_only"
        )
        return config


class SPATIAL_MAP_REPORTING_PIPELINE(SPATIAL_MAP_PIPELINE):
    """This method is used to define an eval pipeline with only a metric report component,
    on the spatial map dataset."""

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None) -> PipelineConfig:
        super().configure_pipeline(model_config, resume_from)
        self.evalreporting_comp.data_reader_config.init_args["path"] = resume_from
        # Configure the pipeline
        return PipelineConfig(
            [
                self.preeval_data_post_processing_comp,
                self.filter_empty_answer,
                self.inference_llm_answer_extract,
                self.data_join,                
                self.evalreporting_comp,
                self.posteval_data_post_processing_comp,
                self.bon_evalreporting_comp,
                self.won_evalreporting_comp,
                self.data_post_processing_mv,
                self.mv_evalreporting_comp
            ],
            self.log_dir,
        )
