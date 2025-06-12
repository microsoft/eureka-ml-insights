"""This file contains an implementation of the NoCaps eval.
"""

import os
from typing import Any

from eureka_ml_insights.core import (
    EvalReporting,
    Inference,
    PromptProcessing
)
from eureka_ml_insights.data_utils import (
    AddColumnAndData,
    AddColumn,
    CopyColumn,
    ColumnRename,
    DataReader,
    HFDataReader,
    MapStringsTransform,
    MMDataLoader,
    SamplerTransform,
    SequenceTransform,
)

from eureka_ml_insights.configs import(
    AggregatorConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)


from eureka_ml_insights.metrics.reports import AverageAggregator, ValueFilteredAggregator
from eureka_ml_insights.configs import ExperimentConfig
from eureka_ml_insights.configs.model_configs import OAI_GPT4_1106_PREVIEW_CONFIG as PERSONAL_GPT4O


class NOCAPS_PIPELINE(ExperimentConfig):
    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        # Configure the data processing component.
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "HuggingFaceM4/NoCaps",
                    "split": "validation",
                    "transform": SequenceTransform(
                        [
                            AddColumnAndData(column_name="prompt", data="Write a brief caption to summarize the contents of the image."),
                            SamplerTransform(sample_count=200, random_seed=1234),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
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

        # Eval data pre processing component
        self.eval_data_pre_processing = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform([ColumnRename(name_mapping={"model_output": "response"})]),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/nocaps_templates/scoring_prompt.jinja"
            ),
            output_dir=os.path.join(self.log_dir, "eval_data_pre_processing_output"),
        )

        # Eval Inference component (LLM scoring)
        self.eval_inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=PERSONAL_GPT4O,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {"path": os.path.join(self.eval_data_pre_processing.output_dir, "transformed_data.jsonl"), "load_images":False},
            ),
            output_dir=os.path.join(self.log_dir, "eval_inference_result"),
        )

        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.eval_inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            AddColumn(column_name="score"),
                            CopyColumn(column_name_src="model_output", column_name_dst="score"),
                            MapStringsTransform(columns=["score"], mapping = lambda x: "-1" if not isinstance(x, str) else x.split("SCORE: ")[-1] if x.find("SCORE: ") != -1 else x.split("SCORE ")[-1] if x.find("SCORE ") != -1 else None),
                        ]
                    )
                },
            ),
            aggregator_configs=[
                AggregatorConfig(
                    ValueFilteredAggregator,
                    {
                        "agg_class": AverageAggregator,
                        "value": "-1",
                        "column_names": ["score"],
                        "filename_base": "NoCaps_Score",
                        "ignore_non_numeric": True,
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.eval_data_pre_processing,
                self.eval_inference_comp,
                self.evalreporting_comp,
            ],
            self.log_dir,
        )
