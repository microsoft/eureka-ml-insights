import os
from typing import Any
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    DataReader,
    HFDataReader,
    MMDataLoader,
    SequenceTransform,
    RunPythonTransform,
    CopyColumn,
    RegexTransform,
    SamplerTransform,
    ImputeNA
)
from eureka_ml_insights.metrics import AverageAggregator, MaxTokenF1ScoreMetric
from .config import (
    AggregatorConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from .experiment_config import ExperimentConfig
from eureka_ml_insights.data_utils.drop_utils import CreateDropPrompt

"""This file contains user defined configuration classes for the geometric reasoning task on geometer dataset.
"""
class Drop_Experiment_Pipeline(ExperimentConfig):
    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        # Configure the data processing component.
        data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "ucinlp/drop",
                    "split": "validation",
                    "transform": SequenceTransform(
                        [
                            RunPythonTransform(python_code="df['ground_truth'] = df['answers_spans'].apply(lambda x: x['spans'])")
                        ]
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/drop_templates/basic.jinja"
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        # Configure the inference component
        inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {"path": os.path.join(data_processing_comp.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
        )

        # # Configure the evaluation and reporting component.
        evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            CopyColumn(column_name_src="model_output", column_name_dst="raw_model_output"),
                            RegexTransform(columns="model_output", prompt_pattern=r"My answer is (.+)", case=True),
                            ImputeNA(columns="model_output", value="")
                        ]
                    ),
                },
            ),
            metric_config=MetricConfig(MaxTokenF1ScoreMetric),
            aggregator_configs=[
                AggregatorConfig(AverageAggregator, {"column_names": ["MaxTokenF1ScoreMetric_result"]})
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )
        # # Configure the pipeline
        return PipelineConfig(
            [
                data_processing_comp,
                inference_comp,
                evalreporting_comp,
            ],
            self.log_dir,
        )