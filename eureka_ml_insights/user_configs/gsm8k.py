import json
import os
from typing import Any, Optional

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
from eureka_ml_insights.core import DataProcessing, Inference, PromptProcessing
from eureka_ml_insights.core.eval_reporting import EvalReporting
from eureka_ml_insights.data_utils import (
    AddColumn,
    ColumnRename,
    CopyColumn,
    DataReader,
    HFDataReader,
    MajorityVoteTransform,
    MultiplyTransform,
    ReplaceStringsTransform,
    SamplerTransform,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.data import DataLoader
from eureka_ml_insights.data_utils.gsm8k_utils import GSM8KExtractAnswer
from eureka_ml_insights.metrics.metrics_base import ExactMatch
from eureka_ml_insights.metrics.reports import (
    BiLevelAggregator,
    BiLevelCountAggregator,
    CountAggregator,
)


class GSM8K_PIPELINE(ExperimentConfig):
    """This class specifies the config for running GSM8K benchmark on any model"""

    def configure_pipeline(
        self,
        model_config: ModelConfig,
        resume_from: str = None,
        n_repeats: int = 1,
        **kwargs: dict[str, Any],
    ) -> PipelineConfig:

        # --------------------------------------
        # Data preprocessing
        # --------------------------------------
        # * PromptProcessing:
        # prepare data for inference, apply transformation, or apply a Jinja prompt template.

        self.preprocessing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "openai/gsm8k",
                    "split": "train",
                    "tasks": "main",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(
                                name_mapping={
                                    "question": "prompt",
                                }
                            ),
                            # SamplerTransform(sample_count=5, random_seed=99),
                            MultiplyTransform(n_repeats=int(n_repeats)),
                        ],
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/gsm8k_templates/zeroshot.jinja"
            ),
            output_dir=os.path.join(self.log_dir, "data_preprocessing_output"),
        )

        # --------------------------------------
        # Inference
        # --------------------------------------
        # * Inference:  run your model on any processed data,
        # for example running inference on the model subject to evaluation,
        # or another model that is involved in the evaluation pipeline as an evaluator or judge.
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                DataLoader,
                {"path": os.path.join(self.preprocessing_comp.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(self.log_dir, "data_inference_result"),
            resume_from=resume_from,
            max_concurrent=10,
        )

        # --------------------------------------
        # Extract answer
        # --------------------------------------
        # * DataProcessing: you can use this component to to post-process the model outputs.
        self.postprocessing_comp = DataProcessingConfig(
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
                            AddColumn("ground_truth"),
                            GSM8KExtractAnswer("raw_output", "model_output"),
                            GSM8KExtractAnswer("answer", "ground_truth"),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_postprocessing_output"),
        )

        # --------------------------------------
        # Evaluation (compute accuracy)
        # --------------------------------------
        # * EvalReporting: evaluate the model outputs using various metrics, aggregators
        # and visualizers, and generate a report.
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.postprocessing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(ExactMatch),
            aggregator_configs=[
                # Aggregates across all repeats in one pool (no group_by)
                # - single overall pass@1 score for the entire dataset
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": ["ExactMatch_result"],
                        "normalize": True,
                        "filename_base": "ExactMatch",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        if int(n_repeats) > 1:

            self.evalreporting_comp.aggregator_configs.extend(
                [
                    # Separate run accuracy (pass@1 for each repeat)
                    AggregatorConfig(
                        CountAggregator,
                        {
                            "column_names": ["ExactMatch_result"],
                            "group_by": "data_repeat_id",
                            "filename_base": "ExactMatch_SeparateRuns",
                            "normalize": True,
                        },
                    ),
                    # All-run accuracy (mean and std of pass@1 across repeats)
                    AggregatorConfig(
                        BiLevelCountAggregator,
                        {
                            "column_names": ["ExactMatch_result"],
                            "first_groupby": "data_repeat_id",
                            "filename_base": "ExactMatch_AverageOfRuns",
                            "normalize": True,
                        },
                    ),
                ]
            )

            # --------------------------------------
            # Post-eval best-of-N/worst-of-N
            # --------------------------------------

            # Convert "ExactMatch_result" from correct/incorrect => 1/0
            self.posteval_data_numeric_comp = DataProcessingConfig(
                component_type=DataProcessing,
                data_reader_config=DataSetConfig(
                    DataReader,
                    {
                        "path": os.path.join(self.evalreporting_comp.output_dir, "metric_results.jsonl"),
                        "format": ".jsonl",
                        "transform": SequenceTransform(
                            [
                                CopyColumn("ExactMatch_result", "ExactMatch_result_numeric"),
                                # AddColumn("ExactMatch_result_numeric"),
                                ReplaceStringsTransform(
                                    columns=["ExactMatch_result_numeric"],
                                    mapping={"correct": "1", "incorrect": "0", "none": "NaN"},
                                    case=False,
                                ),
                            ]
                        ),
                    },
                ),
                output_dir=os.path.join(self.log_dir, "data_posteval_numeric_output"),
            )

            # Best-of-n aggregator
            self.bestofn_evalreporting_comp = EvalReportingConfig(
                component_type=EvalReporting,
                data_reader_config=DataSetConfig(
                    DataReader,
                    {
                        "path": os.path.join(self.posteval_data_numeric_comp.output_dir, "transformed_data.jsonl"),
                        "format": ".jsonl",
                    },
                ),
                aggregator_configs=[
                    AggregatorConfig(
                        BiLevelAggregator,
                        {
                            "column_names": ["ExactMatch_result_numeric"],
                            "first_groupby": "data_point_id",
                            "filename_base": "ExactMatch_BestOfN",
                            "agg_fn": "max",
                        },
                    ),
                ],
                output_dir=os.path.join(self.log_dir, "eval_report_bestofn"),
            )

            # Worst-of-n aggregator
            self.worstofn_evalreporting_comp = EvalReportingConfig(
                component_type=EvalReporting,
                data_reader_config=DataSetConfig(
                    DataReader,
                    {
                        "path": os.path.join(self.posteval_data_numeric_comp.output_dir, "transformed_data.jsonl"),
                        "format": ".jsonl",
                    },
                ),
                aggregator_configs=[
                    AggregatorConfig(
                        BiLevelAggregator,
                        {
                            "column_names": ["ExactMatch_result_numeric"],
                            "first_groupby": "data_point_id",
                            "filename_base": "ExactMatch_WorstOfN",
                            "agg_fn": "min",
                        },
                    ),
                ],
                output_dir=os.path.join(self.log_dir, "eval_report_worstofn"),
            )

            # --------------------------------------
            # Majority voting for multiple runs
            # --------------------------------------
            self.postprocessing_majorityvote_comp = DataProcessingConfig(
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
                                GSM8KExtractAnswer("raw_output", "model_output"),
                                GSM8KExtractAnswer("answer", "ground_truth"),
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
                output_dir=os.path.join(self.log_dir, "data_postprocessing_output_majorityvote"),
            )

            self.evalreporting_majorityvote_comp = EvalReportingConfig(
                component_type=EvalReporting,
                data_reader_config=DataSetConfig(
                    DataReader,
                    {
                        "path": os.path.join(
                            self.postprocessing_majorityvote_comp.output_dir, "transformed_data.jsonl"
                        ),
                        "format": ".jsonl",
                    },
                ),
                metric_config=MetricConfig(ExactMatch),
                aggregator_configs=[
                    AggregatorConfig(
                        BiLevelCountAggregator,
                        {
                            "column_names": ["ExactMatch_result"],
                            "first_groupby": "data_point_id",
                            "filename_base": "MajorityVote",
                            "normalize": True,
                        },
                    ),
                ],
                output_dir=os.path.join(self.log_dir, "eval_report_majorityvote"),
            )

        pipeline_steps = [
            self.preprocessing_comp,
            self.inference_comp,
            self.postprocessing_comp,
            self.evalreporting_comp,
        ]

        if int(n_repeats) > 1:
            pipeline_steps.extend(
                [
                    self.posteval_data_numeric_comp,
                    self.bestofn_evalreporting_comp,
                    self.worstofn_evalreporting_comp,
                    self.postprocessing_majorityvote_comp,
                    self.evalreporting_majorityvote_comp,
                ]
            )

        return PipelineConfig(pipeline_steps, self.log_dir)


# =============================
# MUTATED GSM8K BENCHMARK
# =============================


class GSM8K_MUTATED_PIPELINE(GSM8K_PIPELINE):
    """This class specifies the config for running mutated GSM8K benchmark on any model"""

    def configure_pipeline(
        self,
        model_config: ModelConfig,
        resume_from: str = None,
        mutation_type: str = "Factual",
        n_repeats: Optional[int] = 1,
        sample_count: Optional[int] = None,
        prompt_template_name: Optional[str] = None,
        **kwargs: dict[str, Any],
    ) -> PipelineConfig:

        pipeline = super().configure_pipeline(
            model_config=model_config,
            resume_from=resume_from,
            n_repeats=n_repeats,
        )

        # -----------------------------
        # Retrieve data and pre-process
        # -----------------------------
        with open("data/paths_gsm8k.json", "r") as f:
            DATA_PATHS = json.load(f)
        path = DATA_PATHS[mutation_type]

        self.preprocessing_comp.data_reader_config = DataSetConfig(
            HFDataReader,
            {
                "path": path,
                "split": "validation",
                "load_data_from_disk": True,
                "transform": SequenceTransform(
                    [
                        ColumnRename(
                            name_mapping={
                                "question": "prompt",
                                "answer": "ground_truth",
                            }
                        ),
                    ],
                ),
            },
        )

        if sample_count is not None:
            self.preprocessing_comp.data_reader_config.init_args["transform"].transforms.append(
                SamplerTransform(sample_count=int(sample_count), random_seed=99)
            )

        self.preprocessing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=int(n_repeats)),
        )

        if prompt_template_name is not None:
            self.preprocessing_comp.prompt_template_path = os.path.join(
                os.path.dirname(__file__), f"../prompt_templates/gsm8k_templates/{prompt_template_name}.jinja"
            )

        # --------------------------------
        # Post-processing data transform
        # --------------------------------

        self.postprocessing_comp.data_reader_config.init_args["transform"] = SequenceTransform(
            [
                ColumnRename(
                    name_mapping={
                        "model_output": "raw_output",
                    }
                ),
                AddColumn("model_output"),
                GSM8KExtractAnswer("raw_output", "model_output"),
            ]
        )

        if int(n_repeats) > 1:
            self.postprocessing_majorityvote_comp.data_reader_config.init_args["transform"] = SequenceTransform(
                [
                    ColumnRename(name_mapping={"model_output": "raw_output"}),
                    AddColumn("model_output"),
                    GSM8KExtractAnswer("raw_output", "model_output"),
                    MajorityVoteTransform(id_col="data_point_id"),
                    ColumnRename(
                        name_mapping={
                            "model_output": "model_output_onerun",
                            "majority_vote": "model_output",
                        }
                    ),
                ]
            )

        return pipeline
