import os
from typing import Any

from eureka_ml_insights.configs.experiment_config import ExperimentConfig
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    ASTEvalTransform,
    ColumnRename,
    CopyColumn,
    DataReader,
    HFDataReader,
    MapStringsTransform,
    MMDataLoader,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.mmmu_utils import (
    CreateMMMUPrompts,
    MMMUAll,
    MMMUTaskToCategories,
)
from eureka_ml_insights.metrics import CountAggregator, MMMUMetric

from eureka_ml_insights.configs import(
    AggregatorConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)


class MMMU_BASELINE_PIPELINE(ExperimentConfig):
    """
    This defines an ExperimentConfig pipeline for the MMMU dataset.
    There is no model_config by default and the model config must be passed in via command lime.
    """

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any] ) -> PipelineConfig:
    
        self.data_processing_comp = PromptProcessingConfig(
        component_type=PromptProcessing,
        data_reader_config=DataSetConfig(
            HFDataReader,
            {
                "path": "MMMU/MMMU",
                "split": "validation",
                "tasks": ["Art"],
                "transform": SequenceTransform(
                    [
                        ASTEvalTransform(columns=["options"]),
                        CreateMMMUPrompts(),
                        ColumnRename(name_mapping={"answer": "ground_truth", "options": "target_options"}),
                    ]
                ),
            },
        ),
        output_dir=os.path.join(self.log_dir, "data_processing_output"),
        ignore_failure=False,
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
        )

        # Configure the evaluation and reporting component.
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            CopyColumn(column_name_src="__hf_task", column_name_dst="category"),
                            MapStringsTransform(
                                columns=["category"],
                                mapping=MMMUTaskToCategories,
                            ),
                        ]
                    ),
                },
            ),
            metric_config=MetricConfig(MMMUMetric),
            aggregator_configs=[
                AggregatorConfig(CountAggregator, {"column_names": ["MMMUMetric_result"], "normalize": True}),
                AggregatorConfig(
                    CountAggregator,
                    {"column_names": ["MMMUMetric_result"], "group_by": "category", "normalize": True},
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # Configure the pipeline
        return PipelineConfig([self.data_processing_comp, self.inference_comp, self.evalreporting_comp], self.log_dir)
