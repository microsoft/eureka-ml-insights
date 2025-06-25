"""
This module defines pipeline configurations for running the GSM8K benchmark and its variations,
including transformations, inference steps, and various evaluations for different model configurations.
"""

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
    ExtractUsageTransform,
    HFDataReader,
    MajorityVoteTransform,
    MapStringsTransform,
    MultiplyTransform,
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
    """
    Specifies the configuration for running the GSM8K benchmark on any model.

    This class extends ExperimentConfig and sets up the data processing, inference,
    post-processing, and evaluation components for the GSM8K dataset. It also allows
    for multiple runs of the pipeline to enable additional aggregations.
    """

    def configure_pipeline(
        self,
        model_config: ModelConfig,
        resume_from: str = None,
        n_repeats: int = 1,
        **kwargs: dict[str, Any],
    ) -> PipelineConfig:
        """
        Configures the pipeline steps for running the GSM8K benchmark.

        Args:
            model_config (ModelConfig): Configuration for the model to be used during inference.
            resume_from (str, optional): Path to resume from if the pipeline restarts. Defaults to None.
            n_repeats (int, optional): Number of times to repeat the pipeline run. Defaults to 1.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            PipelineConfig: The pipeline configuration containing all steps to run the GSM8K benchmark.
        """

        # --------------------------------------
        # * Data preprocessing
        # --------------------------------------
        # Prepare data for inference, apply transformation, or apply a Jinja prompt template.

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
                            # SamplerTransform(sample_count=3, random_seed=99),
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
        # * Inference
        # --------------------------------------
        # Run model on any processed data

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
        # * Extract answer
        # --------------------------------------
        # Extract answer from raw model output

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
                            ExtractUsageTransform(model_config),  # Get token usage information
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_postprocessing_output"),
        )

        # --------------------------------------
        # * Evaluation
        # --------------------------------------
        # Report metrics and aggregate results

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
                # Get average usage across all data points
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": "data_point_id",
                        "filename_base": "UsageCompletion_Mean",
                        "agg_fn": "mean",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        pipeline_steps = [
            self.preprocessing_comp,
            self.inference_comp,
            self.postprocessing_comp,
            self.evalreporting_comp,
        ]

        if int(n_repeats) > 1:
            multirun_steps = self._configure_multirun_steps()
            pipeline_steps.extend(multirun_steps)

        return PipelineConfig(pipeline_steps, self.log_dir)

    def _configure_multirun_steps(self) -> list[Any]:
        """
        Configures and returns additional aggregator configurations and post-evaluation steps
        required for multi-run analysis.

        This includes aggregator configurations for separate/average runs, best-of-n, worst-of-n,
        usage statistics, and majority voting.

        Returns:
            list[Any]: A list of additional steps to extend the pipeline for multi-run analysis.
        """

        # Extend aggregator configs on the existing evalreporting_comp
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
                # Calculate usage stats per repeat
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": "data_repeat_id",
                        "filename_base": "UsageCompletion_MeanofN",
                        "agg_fn": "mean",
                    },
                ),
                # Sums usage across all repeats for each data point
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": "data_point_id",
                        "filename_base": "UsageCompletion_AllN",
                        "agg_fn": "sum",
                    },
                ),
            ]
        )

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
                            MapStringsTransform(
                                columns=["ExactMatch_result_numeric"],
                                mapping={"correct": "1", "incorrect": "0", "none": "NaN"},
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
                # Measures fraction of data points solved by at least one attempt
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
                # Measures fraction of data points correct on every attempt.
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

        # Majority voting for multiple runs
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
                    "path": os.path.join(self.postprocessing_majorityvote_comp.output_dir, "transformed_data.jsonl"),
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

        return [
            self.posteval_data_numeric_comp,
            self.bestofn_evalreporting_comp,
            self.worstofn_evalreporting_comp,
            self.postprocessing_majorityvote_comp,
            self.evalreporting_majorityvote_comp,
        ]


# =============================
# MUTATED GSM8K BENCHMARK
# =============================


class GSM8K_MUTATED_PIPELINE(GSM8K_PIPELINE):
    """
    Specifies the configuration for running the mutated GSM8K benchmark on any model.

    This class modifies the data loading to use mutated GSM8K datasets and applies optional
    transformations such as sampling and different prompt templates.
    """

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
        """
        Configures the pipeline steps for the mutated GSM8K benchmark.

        Args:
            model_config (ModelConfig): Configuration for the model to be used during inference.
            resume_from (str, optional): Path to resume from if the pipeline restarts. Defaults to None.
            mutation_type (str, optional): The type of data mutation to use. Defaults to "Factual".
            n_repeats (Optional[int], optional): Number of times to repeat the pipeline run. Defaults to 1.
            sample_count (Optional[int], optional): Number of samples to draw from the dataset. Defaults to None.
            prompt_template_name (Optional[str], optional): Name of the prompt template file. Defaults to None.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            PipelineConfig: The pipeline configuration containing all steps to run the mutated GSM8K benchmark.
        """

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
                ExtractUsageTransform(model_config),
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


# =============================
# GSM-SYMBOLIC BENCHMARK
# =============================


class GSMSYMBOLIC_PIPELINE(GSM8K_PIPELINE):
    """
    Specifies the configuration for running the GSM-Symbolic benchmark on any model.

    This class modifies the data reader configuration to use the GSM-Symbolic dataset,
    leveraging the same pipeline structure as GSM8K_PIPELINE.
    """

    def configure_pipeline(self, model_config, resume_from=None, n_repeats=1, **kwargs):
        """
        Configures the pipeline steps for the GSM-Symbolic benchmark.

        Args:
            model_config: Configuration for the model to be used during inference.
            resume_from (str, optional): Path to resume from if the pipeline restarts. Defaults to None.
            n_repeats (int, optional): Number of times to repeat the pipeline run. Defaults to 1.
            **kwargs: Additional keyword arguments.

        Returns:
            PipelineConfig: The pipeline configuration containing all steps to run the GSM-Symbolic benchmark.
        """
        pipeline = super().configure_pipeline(
            model_config,
            resume_from,
            n_repeats,
            **kwargs,
        )

        self.preprocessing_comp.data_reader_config = DataSetConfig(
            HFDataReader,
            {
                "path": "apple/GSM-Symbolic",
                "split": "test",
                "tasks": "main",
                "transform": SequenceTransform(
                    [
                        ColumnRename(
                            name_mapping={
                                "question": "prompt",
                            }
                        ),
                        # SamplerTransform(sample_count=3, random_seed=99),
                        MultiplyTransform(n_repeats=int(n_repeats)),
                    ],
                ),
            },
        )

        return pipeline