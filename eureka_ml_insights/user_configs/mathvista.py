"""Implementation of the MathVista eval.

This module provides an implementation for the MathVista eval:
https://mathvista.github.io/.
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
    """A pipeline configuration for the MathVista evaluation.

    This class extends ExperimentConfig to define a pipeline that processes 
    MathVista data, performs inference, and evaluates the results.
    """

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        """Configure the pipeline for MathVista evaluation.

        Args:
            model_config (ModelConfig): The configuration for the model to be used in inference.
            resume_from (str, optional): Path to a checkpoint to resume from. Defaults to None.
            **kwargs (dict[str, Any]): Additional keyword arguments for pipeline configuration.

        Returns:
            PipelineConfig: The configured pipeline for MathVista evaluation.
        """
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
            model_config=PERSONAL_GPT4O,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {"path": os.path.join(self.eval_data_pre_processing.output_dir, "transformed_data.jsonl"), "load_images":False},
            ),
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
            model_config=PERSONAL_GPT4O,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {"path": os.path.join(self.eval_data_pre_processing_two.output_dir, "transformed_data.jsonl"), "load_images":False},
            ),
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