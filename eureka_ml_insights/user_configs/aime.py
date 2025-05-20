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
    MultiplyTransform,
    RegexTransform,
    ReplaceStringsTransform,
    RunPythonTransform,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.aime_utils import AIMEExtractAnswer
from eureka_ml_insights.data_utils.data import DataLoader
from eureka_ml_insights.metrics.aime_metrics import NumericMatch
from eureka_ml_insights.metrics.reports import (
    BiLevelAggregator,
    BiLevelCountAggregator,
    CountAggregator,
)

from .llm_extraction import LLM_EXTRACTION_SUBPIPELINE_MIXIN

# from eureka_ml_insights.data_utils.transform import MajorityVoteTransform


class AIME_PIPELINE(ExperimentConfig):
    """This class specifies the config for running AIME benchmark on any model"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        self.n_repeats = int(kwargs.get('n_repeat', 1))  # Default value is 1
        self.max_concurrent = int(kwargs.get('max_concurrent', 1))  # Default value is 1
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
                            MultiplyTransform(n_repeats=self.n_repeats),
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
                            AIMEExtractAnswer("model_output","extracted_answer"),
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
        metric_config=MetricConfig(NumericMatch, {"model_output_col": answer_col})
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
                                column_name_src="NumericMatch_result",
                                column_name_dst="NumericMatch_result_numeric",
                            ),
                            ReplaceStringsTransform(
                                columns=["NumericMatch_result_numeric"],
                                mapping={"incorrect": "0", "correct": "1", "none": "NaN"},
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
                        "column_names": ["NumericMatch_result_numeric"],
                        "first_groupby": "ID",
                        "filename_base": "NumericMatch_BestOfN",
                        "agg_fn": "max",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["NumericMatch_result_numeric"],
                        "first_groupby": "ID",
                        "second_groupby": "Year",
                        "filename_base": "NumericMatch_BestOfN_GroupBy_Year",
                        "agg_fn": "max",
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
                        "column_names": ["NumericMatch_result_numeric"],
                        "first_groupby": "ID",
                        "filename_base": "NumericMatch_WorstOfN",
                        "agg_fn": "min",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["NumericMatch_result_numeric"],
                        "first_groupby": "ID",
                        "second_groupby": "Year",
                        "filename_base": "NumericMatch_WorstOfN_GroupBy_Year",
                        "agg_fn": "min",
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
                        "column_names": ["usage_completion"],
                        "first_groupby": "ID",
                        "filename_base": "UsageCompletion_WorstOfN",
                        "agg_fn": "mean",
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

class AIME2025_PIPELINE(AIME_PIPELINE):
    """This class specifies the config for running AIME 2025 benchmark"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from,**kwargs)
        self.data_processing_comp.data_reader_config.init_args["path"] = "lchen001/AIME2025"
        return pipeline
    
class AIME_HYBRIDEXTRACT_PIPELINE(AIME_PIPELINE):
    """This class specifies the config for running AIME with a hybrid answer extraction"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from,**kwargs)
        self.llm_extractor_max_concurrent = int(kwargs.get('llm_extractor_max_concurrent', 10))  # Default value is 1
        answer_col = "extracted_answer"
        llm_extraction_subpipeline_conf = LLM_EXTRACTION_SUBPIPELINE_MIXIN()
        self.llm_extraction_subpipeline = llm_extraction_subpipeline_conf.configure_subpipeline(
            extraction_attempt_component=self.answer_extraction_processing,
            extracted_answer_col=answer_col,
            llm_extraction_prompt_template=os.path.join(
                os.path.dirname(__file__),
                "../prompt_templates/aime_templates/extract_aime_answer.jinja",
            ),
            llm_extractor_model_config=OAI_GPT4O_2024_11_20_CONFIG,
            log_dir=self.log_dir,
            llm_extractor_max_concurrent=self.llm_extractor_max_concurrent,
            llm_extractor_answer_transforms=[
                RegexTransform(
                    columns=answer_col, 
                    prompt_pattern = r"-?\d+\.\d+|-?\d+", # match any numeric numbers
                    ignore_case=True,
                ),
            ],
        )

        self.final_preeval_data_processing.data_reader_config.init_args["path"] = os.path.join(
            self.llm_extraction_subpipeline[-1].output_dir, "transformed_data.jsonl")
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.answer_extraction_processing]+
            self.llm_extraction_subpipeline+
            [    self.final_preeval_data_processing,
                self.evalreporting_comp,
                self.data_post_processing_addmv,
                self.mv_evalreporting_comp,
                self.posteval_data_post_processing_comp,
                self.bon_evalreporting_comp,
                self.won_evalreporting_comp
            ],
            self.log_dir,
        )
    
class AIME2025_HYBRIDEXTRACT_PIPELINE(AIME_HYBRIDEXTRACT_PIPELINE):
    """This class specifies the config for running AIME 2025 benchmark"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from,**kwargs)
        self.data_processing_comp.data_reader_config.init_args["path"] = "lchen001/AIME2025"
        return pipeline