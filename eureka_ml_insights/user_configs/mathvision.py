"""Module implementing the MATH-V evaluation logic.

For more details, see: https://github.com/mathllm/MATH-V
"""

import os
from typing import Any

from eureka_ml_insights.configs import (
    AggregatorConfig,
    DataProcessingConfig,
    DataSetConfig,
    EvalReportingConfig,
    ExperimentConfig,
    InferenceConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.core import (
    DataProcessing,
    EvalReporting,
    Inference,
    PromptProcessing,
)
from eureka_ml_insights.data_utils import (
    AddColumn,
    CopyColumn,
    DataReader,
    HFDataReader,
    MapStringsTransform,
    MMDataLoader,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.mathvision_utils import (
    MathVisionOutputEvaluator,
)
from eureka_ml_insights.metrics.reports import AverageAggregator


class MATHVISION_PIPELINE(ExperimentConfig):
    """A pipeline configuration for MATHVISION experiments.

    This class extends from ExperimentConfig and sets up the pipeline
    for data processing, inference, post-processing, and evaluation
    in the MATHVISION scenario.
    """

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        """Configures and returns the pipeline for MATHVISION experiments.

        Args:
            model_config (ModelConfig): The configuration for the model to be used.
            resume_from (str, optional): A path to resume from a previous checkpoint.
                Defaults to None.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            PipelineConfig: The fully configured pipeline.
        """
        # Configure the data processing component.
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "MathLLMs/MathVision",
                    "split": "test",
                    "tasks": ["default"],
                    "transform": SequenceTransform(
                        [
                            CopyColumn(column_name_src="options", column_name_dst="options_string"),
                            MapStringsTransform(
                                columns="options_string",
                                mapping=lambda x: (
                                    ""
                                    if len(x) == 0
                                    else (
                                        "\n[Options]:\n"
                                        + "\n".join([chr(ord("A") + i) + ". " + opt for i, opt in enumerate(x)])
                                    )
                                ),
                            ),
                            # SamplerTransform(sample_count=2, random_seed=1234),
                        ]
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/mathvision_templates/question.jinja"
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

        # post process the response to extract the answer
        self.data_post_processing = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            AddColumn("score"),
                            MathVisionOutputEvaluator(score_column_name="score"),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_post_processing_output"),
        )

        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.data_post_processing.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            aggregator_configs=[
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": ["score"],
                        "filename_base": "MathVision_Score",
                    },
                ),
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": ["score"],
                        "filename_base": "MathVision_Score_By_Type",
                        "group_by": ["level", "subject"],
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.data_post_processing,
                self.evalreporting_comp,
            ],
            self.log_dir,
        )
