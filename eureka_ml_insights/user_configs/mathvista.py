"""This file contains an implementation of the MathVista eval: https://mathvista.github.io/.
"""

import os
from typing import Any

from eureka_ml_insights.core import (
    EvalReporting,
    Inference,
    PromptProcessing
)
from eureka_ml_insights.data_utils import (
    ColumnRename,
    CopyColumn,
    DataReader,
    HFDataReader,
    ImputeNA,
    MapStringsTransform,
    MMDataLoader,
    SamplerTransform,
    SequenceTransform,
    SelectTransform,
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

from eureka_ml_insights.metrics.reports import AverageAggregator
from eureka_ml_insights.configs import ExperimentConfig
from eureka_ml_insights.configs.model_configs import OAI_GPT4_1106_PREVIEW_CONFIG as PERSONAL_GPT4O

class MATHVISTA_PIPELINE(ExperimentConfig):
    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        
        # Get the user provided LLM judge configuration, defaulting to PERSONAL_GPT4O if not provided.
        LLM_JUDGE_CONFIG = kwargs.get("llm_judge_config", PERSONAL_GPT4O)

        # Configure the data processing component.
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "AI4Math/MathVista",
                    "split": "testmini",
                    "tasks": ["default"],
                    "transform": SequenceTransform(
                        [
                            #SelectTransform(indices=list(range(0,1000,10))),  # Select every 10th sample for a smaller subset
                            CopyColumn(column_name_src="query", column_name_dst="prompt"),
                            # there is some info in the metadata field that serves as a nice aggregator
                            CopyColumn(column_name_src="metadata", column_name_dst="task"),
                            MapStringsTransform(columns="task", mapping=lambda d: d["task"]),
                            #SamplerTransform(sample_count=32, random_seed=1234),
                            ImputeNA(columns=["choices", "precision", "unit"], value=""),
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
                {
                    "path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "image_column_names": ["decoded_image"],
                },
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
        )

        # Eval data pre processing component round 1 (answer extraction).
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
                os.path.dirname(__file__), "../prompt_templates/mathvista_templates/answer_extraction_prompt.jinja"
            ),
            output_dir=os.path.join(self.log_dir, "eval_data_pre_processing_output"),
        )

        # Eval Inference component round 1 (answer extraction).
        self.eval_inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=LLM_JUDGE_CONFIG,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {"path": os.path.join(self.eval_data_pre_processing.output_dir, "transformed_data.jsonl"), "load_images":False},
            ),
            max_concurrent=500,
            output_dir=os.path.join(self.log_dir, "eval_inference_result"),
        )

        # Eval data pre processing component round 2 (LLM scoring).
        self.eval_data_pre_processing_two = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.eval_inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform([ColumnRename(name_mapping={"model_output": "extraction"})]),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/mathvista_templates/scoring_prompt.jinja"
            ),
            output_dir=os.path.join(self.log_dir, "eval_data_pre_processing_output_two"),
        )

        # Eval Inference component round 2 (LLM scoring)
        self.eval_inference_comp_two = InferenceConfig(
            component_type=Inference,
            model_config=LLM_JUDGE_CONFIG,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {"path": os.path.join(self.eval_data_pre_processing_two.output_dir, "transformed_data.jsonl"), "load_images":False},
            ),
            max_concurrent=500,
            output_dir=os.path.join(self.log_dir, "eval_inference_result_two"),
        )

        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.eval_inference_comp_two.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": ColumnRename(name_mapping={"model_output": "score"}),
                },
            ),
            aggregator_configs=[
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": ["score"],
                        "filename_base": "MathVista_Score",
                    },
                ),
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": ["score"],
                        "filename_base": "MathVista_Score_By_Type",
                        "group_by": ["question_type", "task"],
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
                self.eval_data_pre_processing_two,
                self.eval_inference_comp_two,
                self.evalreporting_comp,
            ],
            self.log_dir,
        )

class MATHVISTA_REPORTING_PIPELINE(MATHVISTA_PIPELINE):
    """This method is used to define an eval pipeline with only a metric report component,
    on the MathVista dataset."""

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None) -> PipelineConfig:
        super().configure_pipeline(model_config, resume_from)
        self.evalreporting_comp.data_reader_config.init_args["path"] = resume_from

        return PipelineConfig(
            [
                self.evalreporting_comp,
            ],
            self.log_dir,
        )