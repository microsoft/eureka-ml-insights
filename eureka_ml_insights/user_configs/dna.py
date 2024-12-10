import os
from typing import Any, Optional

from eureka_ml_insights.core import (
    DataProcessing,
    Inference,
    PromptProcessing,
)
from eureka_ml_insights.core.eval_reporting import EvalReporting
from eureka_ml_insights.data_utils import (
    ColumnRename,
    HFDataReader,
    MMDataLoader,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.data import DataReader
from eureka_ml_insights.data_utils.dna_utils import DNAParseLabel
from eureka_ml_insights.data_utils.transform import AddColumn
from eureka_ml_insights.metrics.reports import CountAggregator

from eureka_ml_insights.configs import (
    AggregatorConfig,
    DataProcessingConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.configs import ExperimentConfig
from eureka_ml_insights.configs.model_configs import OAI_GPT4_1106_PREVIEW_CONFIG


class DNA_PIPELINE(ExperimentConfig):
    """This class specifies the config for running Do-Not-Answer benchmark on any model"""

    def configure_pipeline(self, model_config: ModelConfig,
                           resume_from: str = None, 
                           **kwargs: dict[str, Any]) -> PipelineConfig:
   
        # data preprocessing
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "LibrAI/do-not-answer",
                    "split": "train",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={"question": "prompt"}),
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
                MMDataLoader,
                {"path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
        )

        # eval data processing
        self.eval_data_pre_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform([ColumnRename(name_mapping={"prompt": "context"})]),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/dna_templates/gpt4_eval.jinja"
            ),
            output_dir=os.path.join(self.log_dir, "eval_data_pre_processing_output"),
        )

        # eval Inference component
        self.eval_inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=OAI_GPT4_1106_PREVIEW_CONFIG,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {"path": os.path.join(self.eval_data_pre_processing_comp.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(self.log_dir, "eval_inference_result"),
        )

        # Configure the data post processing component to extract label
        self.eval_post_processing_comp = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.eval_inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            AddColumn("model_action_label"),
                            AddColumn("model_harmless_label"),
                            DNAParseLabel(
                                model_output_column="model_output",
                                model_action_label_column="model_action_label",
                                model_harmless_label_column="model_harmless_label",
                                use_updated_metric=True,
                            ),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "eval_post_processing_output"),
        )

        # Configure the evaluation and reporting component.
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.eval_post_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            aggregator_configs=[
                AggregatorConfig(CountAggregator, {"column_names": ["model_action_label"], "normalize": True}),
                AggregatorConfig(CountAggregator, {"column_names": ["model_action_label"], "group_by": "risk_area"}),
                AggregatorConfig(CountAggregator, {"column_names": ["model_harmless_label"], "normalize": True}),
                AggregatorConfig(CountAggregator, {"column_names": ["model_harmless_label"], "group_by": "risk_area"}),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # Configure the pipeline
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.eval_data_pre_processing_comp,
                self.eval_inference_comp,
                self.eval_post_processing_comp,
                self.evalreporting_comp,
            ],
            self.log_dir,
        )
