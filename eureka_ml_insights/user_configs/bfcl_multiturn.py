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
from eureka_ml_insights.metrics.bfcl_metrics import BFCLMultiturnMatch
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
DEFAULT_MAX_TURN = 1


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
    "original_prompt",
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
    """This class specifies the config for running BFCL multi-turn benchmark on any model"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        self.n_repeats = int(kwargs.get('n_repeat', 3))  # Default value is 3
        self.max_concurrent = int(kwargs.get('max_concurrent', 10))  # Default value is 10

        # 1. Start of the multi-turn eval       
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
#                            RunPythonTransform(python_code="df = df.assign(prompt=df['prompt'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x))"),
                             #RunPythonTransform(python_code="df = df.assign(prompt=df['prompt'].apply(lambda x: ast.literal_eval(x) ))"),
                             RunPythonTransform(python_code="df['prompt']=df['prompt'].apply(ast.literal_eval)"),
                             RunPythonTransform(python_code="df['original_prompt']=df['prompt']"),

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
        max_user_turn = kwargs.get("max_user_turn", DEFAULT_MAX_TURN)
        max_user_turn = int(max_user_turn)
        # Uncomment if you want to sample a subset of the data for debugging
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            SamplerTransform(sample_count=10, random_seed=42)
        )

        # execuate one user turn
        def next_user_turn(user_turn):
            for i in range(1, n_iter + 1):
                # Student inference component, reads prompts from the last prompt processing component
                last_prompt_proc_comp = component_configs[-1]
                if(user_turn>1):
                    last_prompt_proc_comp = self.hint_processing_comp
                self.toolcall_generation_comp = InferenceConfig(
                    component_type=Inference,
                    model_config=model_config,
                    data_loader_config=DataSetConfig(
                        MMDataLoader,
                        {
                            "path": os.path.join(last_prompt_proc_comp.output_dir, "transformed_data.jsonl"),
                            # if this is not the first iteration, we need to add the previous messages to the data loader config
                            "misc_columns": ["previous_messages"] if (i > 1 or user_turn > 1) else None,
                        },
                    ),
                    output_dir=os.path.join(self.log_dir, f"model_inference_result_turn_{user_turn}_{i}"),
                    resume_from=resume_from_dict.get(i, None),
                    chat_mode=True,
                )

                component_configs.append(self.toolcall_generation_comp)

                # Answer extraction and metric-based verification
                self.toolcall_execution_comp = DataProcessingConfig(
                    component_type=DataProcessing,
                    data_reader_config=DataSetConfig(
                        DataReader,
                        {
                            "path": os.path.join(self.toolcall_generation_comp.output_dir, "inference_result.jsonl"),
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
                    output_dir=os.path.join(self.log_dir, f"toolcall_execution_{user_turn}_{i}"),
                )
                component_configs.append(self.toolcall_execution_comp)

                # Variable maintaining link to the most recent inference result results to be used for evaluation
                # This will be updated to point to the concatenation of results from all iterations

                if i > 1:
                    self.last_inference_result_join_comp = DataUnionConfig(
                        component_type=DataUnion,
                        data_reader_config=DataSetConfig(
                            DataReader,
                            {
                                "path": os.path.join(self.toolcall_execution_comp.output_dir, "transformed_data.jsonl"),
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
                        output_dir=os.path.join(self.log_dir, f"last_inference_result_join_{user_turn}_{i}"),
                    )
                    last_agg_dir = self.last_inference_result_join_comp.output_dir
                    component_configs.append(self.last_inference_result_join_comp)
                else:
                    last_agg_dir = self.toolcall_execution_comp.output_dir

                # Filtering out the rows with correct answer
                self.filtering_comp = DataProcessingConfig(
                    component_type=DataProcessing,
                    data_reader_config=DataSetConfig(
                        DataReader,
                        {
                            "path": os.path.join(self.toolcall_execution_comp.output_dir, "transformed_data.jsonl"),
                            "format": ".jsonl",
                            "transform": RunPythonTransform(python_code="df = df[df['verification_result'] != 'correct']"), # TODO: clean this for filtering. Either remove it, or modify it using the stopping criteria.
                        },
                    ),
                    output_dir=os.path.join(self.log_dir, f"filtering_{user_turn}_{i}"),
                )
                component_configs.append(self.filtering_comp)

                # Create a new prompt to ask the teacher model to provide hints.
                print(f"*********************** i is {i}")
                if(i < n_iter): # still within the current user turn, ask for more follow-up.
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
                        output_dir=os.path.join(self.log_dir, f"hint_processing_output_{user_turn}_{i}"),
                    )
                else:
                    print("**********i not < n_iter**************")
                    self.hint_processing_comp = PromptProcessingConfig(
                        component_type=PromptProcessing,
                        data_reader_config=DataSetConfig(
                            DataReader,
                            {
                                "path": os.path.join(self.filtering_comp.output_dir, "transformed_data.jsonl"),
                                "format": ".jsonl",
                                "transform": RunPythonTransform(python_code=f"df['user_turn'] = {user_turn}"), 

                            },
                        ),
                        prompt_template_path=os.path.join(
                            os.path.dirname(__file__), "../prompt_templates/bfcl_templates/Template_next_user_turn.jinja"
                        ),
                        output_dir=os.path.join(self.log_dir, f"hint_processing_output_{user_turn}_{i}"),
                    )
                component_configs.append(self.hint_processing_comp)

        # 2. Perform inference for each user turn
        component_configs = [self.data_processing_comp]
        for i in range(1,max_user_turn+1):
            next_user_turn(user_turn=i) # step 1/3: the next turn
            # step 2/3: update the completed queries
            if(i>1):
                self.completed_comp = DataUnionConfig(
                    component_type=DataUnion,
                    data_reader_config=DataSetConfig(
                        DataReader,
                        {
                            "path": os.path.join(self.toolcall_execution_comp.output_dir, "transformed_data.jsonl"),
                            "transform": RunPythonTransform(f"df = df[df['num_user_turn'] == {i}]"),

                            "format": ".jsonl",
                        },
                    ),
                    other_data_reader_config=DataSetConfig(
                        DataReader,
                        {
                            "path": os.path.join(lastcompleted_comp_dir, "transformed_data.jsonl"),
                            "format": ".jsonl",
                        },
                    ),
                    output_data_columns=RESULT_COLS,
                    dedupe_cols=["id", "attempt_id"],
                    output_dir=os.path.join(self.log_dir, f"completed_comp_{i}"),
                )
            else:
                self.completed_comp = DataProcessingConfig(
                component_type=DataProcessing,
                data_reader_config=DataSetConfig(
                    DataReader,
                    {
                        "path": os.path.join(self.toolcall_execution_comp.output_dir, "transformed_data.jsonl"),
                        "format": ".jsonl",
                        "transform": RunPythonTransform(f"df = df[df['num_user_turn'] == {i}]"),
                    },
                ),
                output_dir=os.path.join(self.log_dir, f"completed_comp_{i}"),
            )
            lastcompleted_comp_dir = self.completed_comp.output_dir
            component_configs.append(self.completed_comp)

            # step 3/3: only keep the queries that require more than (i+1) user turns
            self.next_turn_comp = DataProcessingConfig(
                component_type=DataProcessing,
                data_reader_config=DataSetConfig(
                    DataReader,
                    {
                        "path": os.path.join(self.toolcall_generation_comp.output_dir, "inference_result.jsonl"),
                        "format": ".jsonl",
                        "transform": RunPythonTransform(f"df = df[df['num_user_turn'] > {min(i,max_user_turn-1)}]"),
                    },
                ),
                output_dir=os.path.join(self.log_dir, f"next_turn_comp_{i}"),
            )
            component_configs.append(self.next_turn_comp)

        # TODOusage extraction needs fix;
        last_agg_dir = self.toolcall_execution_comp.output_dir
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
                            BFCLMultiturnExtractAnswer("previous_messages","extracted_answer",n_iter),
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


# TODO: How to load prompt as a list so that different round of prompts can be used. Done. 

# TODO: Change the first message for each user round so that each time a new user message is used.
# currently, each first turn message is using last_component, which is always hint_processing_comp.
# This is not working for multi-user-turn. In multi user turn, we should add a new last_component, 
# which should be the prompt[i] for the i-th turn.
# also, we should keep the previous message for all the further turns.
# we can just build on top of hint_processing_comp, by replacing the prompt. Put it at the outside of the next_turn.
# that is, the beginning of the turn. Done.

# TODO: figure out the multi-turn eval applied on the first one or two turns. Sometimes it is very strange. d