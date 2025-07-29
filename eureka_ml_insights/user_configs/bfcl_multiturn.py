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
from eureka_ml_insights.configs.model_configs import (
    OAI_GPT4O_2024_11_20_CONFIG,
)
from eureka_ml_insights.core import DataProcessing, Inference, PromptProcessing
from eureka_ml_insights.core.eval_reporting import EvalReporting
from eureka_ml_insights.data_utils import (
    ColumnRename,
    CopyColumn,
    DataReader,
    ExtractUsageTransform,
    HFDataReader,
    ImputeNA,
    MajorityVoteTransform,
    MajorityVoteListTransform,
    MultiplyTransform,
    RegexTransform,
    ReplaceStringsTransform,
    RunPythonTransform,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.bfcl_utils import BFCLExtractAnswer, BFCLMultiturnExtractAnswer
from eureka_ml_insights.data_utils.data import DataLoader
from eureka_ml_insights.metrics.bfcl_metrics import DictMatch, BFCLMultiturnMatch
from eureka_ml_insights.metrics.reports import (
    BiLevelAggregator,
    BiLevelCountAggregator,
    CountAggregator,
)

from .llm_extraction import LLM_EXTRACTION_SUBPIPELINE_MIXIN
from eureka_ml_insights.data_utils.data import MMDataLoader

# from eureka_ml_insights.data_utils.transform import MajorityVoteTransform
from eureka_ml_insights.data_utils.bfcl_multiturn_utils import BFCLMultiturnExecuteCall
from eureka_ml_insights.core import (
    DataProcessing,
    DataUnion,
    Inference,
    PromptProcessing,
)
DEFAULT_N_ITER = 1

resume_from_dict = {}

from eureka_ml_insights.metrics.metrics_base import (
    ExactMatch,
    MetricBasedVerifier,
)

from eureka_ml_insights.data_utils import (
    AddColumnAndData,
    ColumnRename,
    CopyColumn,
    DataReader,
    RunPythonTransform,
    SamplerTransform,
    SequenceTransform,
)

from eureka_ml_insights.core import (
    DataProcessing,
    DataUnion,
    Inference,
    PromptProcessing,
)
from eureka_ml_insights.configs import (
    DataProcessingConfig,
    DataSetConfig,
    DataUnionConfig,
    InferenceConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)

RESULT_COLS = [
    "attempt_id",
    "model_output",
    "uid",
    "prompt",
    "ground_truth",
    "Year",
    "Part",
    "id",
    "extracted_answer",
    "verification_result",
    "usage",
    "previous_messages",
    "initial_config",
    "involved_classes",
    
]

class BFCL_MULTITURN_PIPELINE(ExperimentConfig):
    """This class specifies the config for running AIME benchmark on any model"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        self.n_repeats = int(kwargs.get('n_repeat', 5))  # Default value is 1
        self.max_concurrent = int(kwargs.get('max_concurrent', 10))  # Default value is 1
        
        # data preprocessing
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "lchen001/BFCL_Multiturn",
                    "split": "train",
                    "transform": SequenceTransform(
                        [
                            MultiplyTransform(n_repeats=self.n_repeats),
                            ColumnRename(
                                name_mapping={
                                    "question": "prompt",
                                }
                            ),
                            #SamplerTransform(sample_count=100, random_seed=42)

                            ],
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/bfcl_templates/Template_multi_turn.jinja"
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        n_iter = kwargs.get("n_iter", DEFAULT_N_ITER)
        n_iter = int(n_iter)
        # Uncomment if you want to sample a subset of the data for debugging
        #self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
        #    SamplerTransform(sample_count=2, random_seed=42)
        #)
        component_configs = [self.data_processing_comp]
        for i in range(1, n_iter + 1):
            # Student inference component, reads prompts from the last prompt processing component
            last_prompt_proc_comp = component_configs[-1]
            self.student_inference_comp = InferenceConfig(
                component_type=Inference,
                model_config=model_config,
                data_loader_config=DataSetConfig(
                    MMDataLoader,
                    {
                        "path": os.path.join(last_prompt_proc_comp.output_dir, "transformed_data.jsonl"),
                        # if this is not the first iteration, we need to add the previous messages to the data loader config
                        "misc_columns": ["previous_messages"] if i > 1 else None,
                    },
                ),
                output_dir=os.path.join(self.log_dir, f"model_inference_result_turn_{i}"),
                resume_from=resume_from_dict.get(i, None),
                chat_mode=True,
            )

            component_configs.append(self.student_inference_comp)

            # Answer extraction and metric-based verification
            self.verification_comp = DataProcessingConfig(
                component_type=DataProcessing,
                data_reader_config=DataSetConfig(
                    DataReader,
                    {
                        "path": os.path.join(self.student_inference_comp.output_dir, "inference_result.jsonl"),
                        "format": ".jsonl",
                        "transform": SequenceTransform(
                            [
                                # extract and verify the student answer
                                BFCLMultiturnExecuteCall(f"model_output", f"tool_output"),
                                MetricBasedVerifier(ExactMatch, f"tool_output"),
                                AddColumnAndData("attempt_id", i),
                                CopyColumn(column_name_src="model_output", column_name_dst=f"student_output"),
                            ]
                        ),
                    },
                ),
                output_dir=os.path.join(self.log_dir, f"verification_{i}"),
            )
            component_configs.append(self.verification_comp)

            # Variable maintaining link to the most recent inference result results to be used for evaluation
            # This will be updated to point to the concatenation of results from all iterations

            if i > 1:
                self.last_inference_result_join_comp = DataUnionConfig(
                    component_type=DataUnion,
                    data_reader_config=DataSetConfig(
                        DataReader,
                        {
                            "path": os.path.join(self.verification_comp.output_dir, "transformed_data.jsonl"),
                            "format": ".jsonl",
                        },
                    ),
                    other_data_reader_config=DataSetConfig(
                        DataReader,
                        {
                            "path": os.path.join(last_agg_dir, "transformed_data.jsonl"),
                            "format": ".jsonl",
                        },
                    ),
                    output_data_columns=RESULT_COLS,
                    dedupe_cols=["uid", "attempt_id"],
                    output_dir=os.path.join(self.log_dir, f"last_inference_result_join_{i}"),
                )
                last_agg_dir = self.last_inference_result_join_comp.output_dir
                component_configs.append(self.last_inference_result_join_comp)
            else:
                last_agg_dir = self.verification_comp.output_dir

            # Filtering out the rows with correct answer
            self.filtering_comp = DataProcessingConfig(
                component_type=DataProcessing,
                data_reader_config=DataSetConfig(
                    DataReader,
                    {
                        "path": os.path.join(self.verification_comp.output_dir, "transformed_data.jsonl"),
                        "format": ".jsonl",
                        "transform": RunPythonTransform(python_code="df = df[df['verification_result'] != 'correct']"), # TODO: clean this for filtering. Either remove it, or modify it using the stopping criteria.
                    },
                ),
                output_dir=os.path.join(self.log_dir, f"filtering_{i}"),
            )
            component_configs.append(self.filtering_comp)

            # Create a new prompt to ask the teacher model to provide hints.
            self.hint_processing_comp = PromptProcessingConfig(
                component_type=PromptProcessing,
                data_reader_config=DataSetConfig(
                    DataReader,
                    {
                        "path": os.path.join(self.filtering_comp.output_dir, "transformed_data.jsonl"),
                        "format": ".jsonl",
                    },
                ),
                prompt_template_path=os.path.join(
                    os.path.dirname(__file__), "../prompt_templates/bfcl_templates/hint_creation.jinja"
                ),
                output_dir=os.path.join(self.log_dir, f"hint_processing_output_{i}"),
            )
            component_configs.append(self.hint_processing_comp)

            '''
            # Inference component to ask teacher model to provide hints
            self.teacher_inference_comp = InferenceConfig(
                component_type=Inference,
                model_config=model_config,
                data_loader_config=DataSetConfig(
                    MMDataLoader,
                    {
                        "path": os.path.join(self.hint_processing_comp.output_dir, "transformed_data.jsonl"),
                        "misc_columns": ["previous_messages"],
                    },
                ),
                output_dir=os.path.join(self.log_dir, f"teacher_inference_result_{i}"),
                max_concurrent=10,
                chat_mode=False,
            )
            component_configs.append(self.teacher_inference_comp)
            
            # Prompt processing to ask the stundent to try again
            self.prompt_processing_with_hint = PromptProcessingConfig(
                component_type=PromptProcessing,
                data_reader_config=DataSetConfig(
                    DataReader,
                    {
                        "path": os.path.join(self.teacher_inference_comp.output_dir, "inference_result.jsonl"),
                        "format": ".jsonl",
                        "transform": ColumnRename(name_mapping={"model_output": "teacher_hint"}),
                    },
                ),
                prompt_template_path=os.path.join(
                    os.path.dirname(__file__), "../prompt_templates/aime_templates/prompt_w_hint.jinja"
                ),
                output_dir=os.path.join(self.log_dir, f"teacher_hint_prompt_{i}"),
            )
            component_configs.append(self.prompt_processing_with_hint)
            '''
        # TODOusage extraction needs fix;
        last_agg_dir = self.verification_comp.output_dir
        self.final_preeval_data_processing_usage = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(last_agg_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform([ExtractUsageTransform(model_config),]),
                },
            ),
            output_dir=os.path.join(self.log_dir, "final_preeval_data_processing_usage_output"),
        )

        self.final_preeval_data_processing = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.final_preeval_data_processing_usage.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            BFCLMultiturnExtractAnswer("previous_messages","extracted_answer"),
                            ImputeNA(columns="extracted_answer", value=""),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "final_preeval_data_processing_output"),
        )
        id_name = "id"
        answer_col = "extracted_answer"
        metric_config=MetricConfig(BFCLMultiturnMatch, {"model_output_col": answer_col})
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.final_preeval_data_processing.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=metric_config,
            aggregator_configs=[
                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": [
                            "BFCLMultiturnMatch_result",
                        ],
                        "first_groupby": "data_repeat_id",
                        "filename_base": "BFCLMultiturnMatch",
                        "normalize": True,
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": id_name,
                        "filename_base": "UsageCompletion",
                        "agg_fn": "mean",
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
                    "path": os.path.join(self.final_preeval_data_processing.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            MajorityVoteListTransform(id_col="data_point_id",model_output_col=answer_col),
                            ColumnRename(
                                name_mapping={
                                    answer_col: "model_output_onerun",
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
            metric_config=MetricConfig(BFCLMultiturnMatch),
            aggregator_configs=[
                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": [
                            "BFCLMultiturnMatch_result",
                        ],
                        "first_groupby": "data_repeat_id",
                        "filename_base": "MajorityVote",
                        "normalize": True,
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": id_name,
                        "filename_base": "UsageCompletion",
                        "agg_fn": "mean",
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
                                column_name_src="BFCLMultiturnMatch_result",
                                column_name_dst="BFCLMultiturnMatch_result_numeric",
                            ),
                            ReplaceStringsTransform(
                                columns=["BFCLMultiturnMatch_result_numeric"],
                                mapping={"true": "1", "false": "0", "none": "NaN"},
                                case=False,
                            ),
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
                    "format": ".jsonl",
                },
            ),
            aggregator_configs=[
                # the first three reports aggregate results by ID and take the best out of N
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["BFCLMultiturnMatch_result_numeric"],
                        "first_groupby": id_name,
                        "filename_base": "BFCLMultiturnMatch_BestOfN",
                        "agg_fn": "max",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": id_name,
                        "filename_base": "UsageCompletion_BestOfN",
                        "agg_fn": "mean",
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
                    "format": ".jsonl",
                },
            ),
            aggregator_configs=[
                # the first three reports aggregate results by ID and take the best out of N
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["BFCLMultiturnMatch_result_numeric"],
                        "first_groupby": id_name,
                        "filename_base": "BFCLMultiturnMatch_WorstOfN",
                        "agg_fn": "min",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": id_name,
                        "filename_base": "UsageCompletion_WorstOfN",
                        "agg_fn": "mean",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "worstofn_eval_report"),
        )
        '''
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
            max_concurrent=self.max_concurrent,
        )
        # post process the response to extract the answer
        self.answer_extraction_processing = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            BFCLExtractAnswer("model_output","extracted_answer"),
                            ImputeNA(columns="extracted_answer", value=""),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "answer_extraction_processing_output"),
        )
        self.final_preeval_data_processing = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.answer_extraction_processing.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform([ExtractUsageTransform(model_config),]),
                },
            ),
            output_dir=os.path.join(self.log_dir, "final_preeval_data_processing_output"),
        )

        # Configure the evaluation and reporting component for evaluation and dataset level aggregation
        answer_col = "extracted_answer"
        metric_config=MetricConfig(DictMatch, {"model_output_col": answer_col})
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.final_preeval_data_processing.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=metric_config,
            aggregator_configs=[
                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": [
                            "DictMatch_result",
                        ],
                        "first_groupby": "data_repeat_id",
                        "filename_base": "DictMatch",
                        "normalize": True,
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": "ID",
                        "filename_base": "UsageCompletion",
                        "agg_fn": "mean",
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
                    "path": os.path.join(self.final_preeval_data_processing.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            MajorityVoteTransform(id_col="ID",model_output_col=answer_col),
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
            metric_config=MetricConfig(DictMatch),
            aggregator_configs=[
                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": [
                            "DictMatch_result",
                        ],
                        "first_groupby": "data_repeat_id",
                        "filename_base": "MajorityVote",
                        "normalize": True,
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": "ID",
                        "filename_base": "UsageCompletion",
                        "agg_fn": "mean",
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
                                column_name_src="DictMatch_result",
                                column_name_dst="DictMatch_result_numeric",
                            ),
                            ReplaceStringsTransform(
                                columns=["DictMatch_result_numeric"],
                                mapping={"true": "1", "false": "0", "none": "NaN"},
                                case=False,
                            ),
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
                    "format": ".jsonl",
                },
            ),
            aggregator_configs=[
                # the first three reports aggregate results by ID and take the best out of N
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["DictMatch_result_numeric"],
                        "first_groupby": "ID",
                        "filename_base": "NumericMatch_BestOfN",
                        "agg_fn": "max",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": "ID",
                        "filename_base": "UsageCompletion_BestOfN",
                        "agg_fn": "mean",
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
                    "format": ".jsonl",
                },
            ),
            aggregator_configs=[
                # the first three reports aggregate results by ID and take the best out of N
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["DictMatch_result_numeric"],
                        "first_groupby": "ID",
                        "filename_base": "NumericMatch_WorstOfN",
                        "agg_fn": "min",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": "ID",
                        "filename_base": "UsageCompletion_WorstOfN",
                        "agg_fn": "mean",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "worstofn_eval_report"),
        )
        
        component_configs.extend(
            [
                self.final_preeval_data_processing,
                self.evalreporting_comp,
                self.data_post_processing_addmv,
                self.mv_evalreporting_comp,
                self.posteval_data_post_processing_comp,
                self.bon_evalreporting_comp,
                self.won_evalreporting_comp,
            ]
        )
        '''
        component_configs.extend(
            [
                self.final_preeval_data_processing_usage,
                self.final_preeval_data_processing,
                self.evalreporting_comp,
                self.data_post_processing_addmv,
                self.mv_evalreporting_comp,
                self.posteval_data_post_processing_comp,
                self.bon_evalreporting_comp,
                self.won_evalreporting_comp,
            ]
        )
        # Configure the pipeline
        return PipelineConfig(
            component_configs,
            self.log_dir,
        )

        # Configure the pipeline
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.answer_extraction_processing,
                self.final_preeval_data_processing,
                self.evalreporting_comp,
                self.data_post_processing_addmv,
                self.mv_evalreporting_comp,
                self.posteval_data_post_processing_comp,
                self.bon_evalreporting_comp,
                self.won_evalreporting_comp,
            ],
            self.log_dir,
        )
