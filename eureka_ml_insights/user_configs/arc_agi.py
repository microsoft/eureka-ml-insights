import os
from typing import Any

from eureka_ml_insights.core import Inference, PromptProcessing
from eureka_ml_insights.core.data_processing import DataProcessing
from eureka_ml_insights.core.eval_reporting import EvalReporting
from eureka_ml_insights.data_utils.arc_agi_utils import ARCAGI_ExtractAnswer
from eureka_ml_insights.data_utils.data import (
    DataLoader,
    DataReader,
    HFDataReader,
)
from eureka_ml_insights.data_utils.transform import (
    AddColumn,
    CleanCOTAnswer,
    ColumnRename,
    CopyColumn,
    ExtractUsageTransform,
    MultiplyTransform,
    ReplaceStringsTransform,
    SequenceTransform,
)
from eureka_ml_insights.metrics.metrics_base import ExactMatch
from eureka_ml_insights.metrics.reports import (
    BiLevelAggregator,
    CountAggregator,
)

from ..configs.config import (
    AggregatorConfig,
    DataProcessingConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from ..configs.experiment_config import ExperimentConfig


class ARC_AGI_v1_PIPELINE(ExperimentConfig):
    """This class specifies the config for running the ARC-AGI-v1 benchmark on any model"""

    def configure_pipeline(self, model_config=None, resume_from=None, **kwargs) -> PipelineConfig:
        # data preprocessing
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/arc_agi_templates/arc_agi_v1_grid_explanation.jinja"
            ),
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "pxferna/ARC-AGI-v1",
                    "split": "test",
                    "transform": SequenceTransform(
                        [
                            MultiplyTransform(n_repeats=1),
                        ]
                    ),
                },
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
            max_concurrent=16,
        )

        # Configure the data post processing component.
        # We keep this component empty so that there is a place to insert a post-processing transform
        # to strip out the chain of thought from models that insert chains of thought using
        # <think> and </think> tags.
        self.data_post_processing = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform([]),
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
                    "transform": SequenceTransform(
                        [
                            ExtractUsageTransform(model_config),
                            ColumnRename(
                                name_mapping={
                                    "model_output": "raw_output",
                                }
                            ),
                            AddColumn("model_output"),
                            ARCAGI_ExtractAnswer("raw_output", "model_output"),
                        ]
                    ),
                },
            ),
            metric_config=MetricConfig(ExactMatch),
            aggregator_configs=[
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": [
                            "ExactMatch_result",
                        ],
                        "filename_base": "OverallMetrics_Grouped_by_Split",
                        "normalize": True,
                        "group_by": "split",
                    },
                ),
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": [
                            "ExactMatch_result",
                        ],
                        "normalize": True,
                        "filename_base": "OverallMetrics_Total",
                    },
                ),
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
                                column_name_src="ExactMatch_result",
                                column_name_dst="ExactMatch_result_numeric",
                            ),
                            ReplaceStringsTransform(
                                columns=["ExactMatch_result_numeric"],
                                mapping={"incorrect": "0", "correct": "1", "none": "NaN"},
                                case=False,
                            ),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "posteval_data_post_processing_output"),
        )

        self.best_of_n_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.posteval_data_post_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            aggregator_configs=[
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": [
                            "ExactMatch_result_numeric",
                        ],
                        "first_groupby": "data_point_id",
                        "filename_base": "ExactMatch_Total_BestOfN",
                        "agg_fn": "max",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["ExactMatch_result_numeric"],
                        "first_groupby": "data_point_id",
                        "second_groupby": "split",
                        "filename_base": "ExactMatch_Grouped_by_Split_BestOfN",
                        "agg_fn": "max",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "bestofn_eval_report"),
        )

        # Configure the pipeline
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.data_post_processing,
                self.evalreporting_comp,
                self.posteval_data_post_processing_comp,
                self.best_of_n_evalreporting_comp,
            ],
            self.log_dir,
        )


class COT_ARC_AGI_v1_PIPELINE(ARC_AGI_v1_PIPELINE):
    """
    This class specifies the config for running the ARC-AGI-v1 benchmark on a reasoning model that
    returns its chain of thought wrapped in <think> and </think> tags.
    """

    def configure_pipeline(self, model_config=None, resume_from=None, **kwargs):
        config = super().configure_pipeline(model_config=model_config, resume_from=resume_from)

        # Here we modify the existing post processing component by inserting a post-processing
        # transform to strip out the chain of thought inserted between <think> and </think> tags.
        self.data_post_processing.data_reader_config.init_args["transform"] = SequenceTransform(
            [
                ColumnRename(
                    name_mapping={
                        "model_output": "cot_model_output",
                    }
                ),
                AddColumn("post_cot_model_output"),
                CleanCOTAnswer("cot_model_output", "post_cot_model_output"),
                CopyColumn("post_cot_model_output", "model_output"),
            ]
        )
        return config


class ARC_AGI_v1_PIPELINE_5Run(ARC_AGI_v1_PIPELINE):
    """
    This class specifies the config for running the ARC-AGI-v1 benchmark 5 repeated times
    on any model.
    """

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=5)
        )
        return pipeline


class COT_ARC_AGI_v1_PIPELINE_5Run(COT_ARC_AGI_v1_PIPELINE):
    """This class specifies the config for running the GPQA benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=5)
        )
        return pipeline


class ARC_AGI_v1_PIPELINE_5050_SUBSET(ARC_AGI_v1_PIPELINE):

    def configure_pipeline(self, model_config=None, resume_from=None, **kwargs):
        config = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        self.data_processing_comp.data_reader_config = DataSetConfig(
            HFDataReader,
            {
                "path": "pxferna/ARC-AGI-v1-5050",
                "split": "train",
                "transform": SequenceTransform(
                    [
                        MultiplyTransform(n_repeats=1),
                    ]
                ),
            },
        )

        return config


class COT_ARC_AGI_v1_PIPELINE_5050_SUBSET(ARC_AGI_v1_PIPELINE_5050_SUBSET):
    """
    This class specifies the config for running The ARC-AGI-v1-5050 subset of the benchmark
    on a reasoning model that returns its chain of thought wrapped in <think> and </think> tags.
    """

    def configure_pipeline(self, model_config=None, resume_from=None, **kwargs):
        config = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        self.data_post_processing.data_reader_config.init_args["transform"] = SequenceTransform(
            [
                ColumnRename(
                    name_mapping={
                        "model_output": "cot_model_output",
                    }
                ),
                AddColumn("post_cot_model_output"),
                CleanCOTAnswer("cot_model_output", "post_cot_model_output"),
                CopyColumn("post_cot_model_output", "model_output"),
            ]
        )
        return config


class COT_ARC_AGI_v1_PIPELINE_5050_SUBSET_5Run(COT_ARC_AGI_v1_PIPELINE_5050_SUBSET):
    """
    This class specifies the config for running The ARC-AGI-v1-5050 subset of the benchmark 5 repeated times
    on a reasoning model that returns its chain of thought wrapped in <think> and </think> tags.
    """

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=5)
        )
        return pipeline
