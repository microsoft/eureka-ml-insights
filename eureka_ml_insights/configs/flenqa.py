import os
from typing import Any, Optional

from eureka_ml_insights.core import (
    DataProcessing,
    EvalReporting,
    Inference,
    PromptProcessing,
)
from eureka_ml_insights.data_utils import (
    ColumnRename,
    DataReader,
    HFDataReader,
    MMDataLoader,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.flenqa_utils import FlenQAOutputProcessor
from eureka_ml_insights.metrics import CountAggregator, ExactMatch

from .config import (
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
from .experiment_config import ExperimentConfig


class FlenQA_Experiment_Pipeline(ExperimentConfig):
    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        # Configure the data pre processing component.
        self.data_pre_processing = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "alonj/FLenQA",
                    "split": ["eval"],
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={"assertion/question": "question", "label": "ground_truth"}),
                        ]
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/flenqa_templates/flenqa.jinja"
            ),
            output_dir=os.path.join(self.log_dir, "data_pre_processing_output"),
        )

        # Inference component
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {"path": os.path.join(self.data_pre_processing.output_dir, "transformed_data.jsonl")},
            ),
            resume_from=resume_from,
            output_dir=os.path.join(self.log_dir, "inference_result"),
        )

        # Configure the data post processing component.
        self.data_post_processing = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "transform": FlenQAOutputProcessor(),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_post_processing_output"),
        )

        # Configure the evaluation and reporting component.
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.data_post_processing.output_dir, "transformed_data.jsonl"),
                    "transform": ColumnRename(
                        name_mapping={"model_output": "raw_model_output", "categorical_response": "model_output"}
                    ),
                },
            ),
            metric_config=MetricConfig(ExactMatch),
            aggregator_configs=[
                AggregatorConfig(CountAggregator, {"column_names": ["ExactMatch_result"], "normalize": True}),
                AggregatorConfig(
                    CountAggregator, {"column_names": ["ExactMatch_result"], "group_by": "ctx_size", "normalize": True}
                ),
                AggregatorConfig(
                    CountAggregator,
                    {"column_names": ["ExactMatch_result"], "group_by": "dataset", "normalize": True},
                ),
                AggregatorConfig(
                    CountAggregator,
                    {"column_names": ["ExactMatch_result"], "group_by": ["ctx_size", "dataset"], "normalize": True},
                ),
                AggregatorConfig(
                    CountAggregator,
                    {"column_names": ["ExactMatch_result"], "group_by": "padding_type", "normalize": True},
                ),
                AggregatorConfig(
                    CountAggregator,
                    {"column_names": ["ExactMatch_result"], "group_by": "dispersion", "normalize": True},
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )
        return PipelineConfig(
            [self.data_pre_processing, self.inference_comp, self.data_post_processing, self.evalreporting_comp],
            self.log_dir,
        )
