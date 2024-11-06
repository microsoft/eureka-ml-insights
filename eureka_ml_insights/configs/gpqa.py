import os
from typing import Any
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    DataReader,
    HFDataReader,
    MMDataLoader,
    SequenceTransform,
    CopyColumn,
    ShuffleColumns,
    ColumnMatchMap,
    SamplerTransform,
    RegexTransform,
    ImputeNA
)
from eureka_ml_insights.metrics import CountAggregator, ExactMatch
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

"""This file contains user defined configuration classes for the geometric reasoning task on the GPQA dataset.
"""
class GPQA_Experiment_Pipeline(ExperimentConfig):
    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        # Configure the data processing component.
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "Idavidrein/gpqa",
                    "tasks": "gpqa_diamond",
                    "split": "train",
                    "transform": SequenceTransform(
                        [
                            #SamplerTransform(sample_count=30, random_seed=42),
                            CopyColumn(column_name_src="Correct Answer", column_name_dst="A"),
                            CopyColumn(column_name_src="Incorrect Answer 1", column_name_dst="B"),
                            CopyColumn(column_name_src="Incorrect Answer 2", column_name_dst="C"),
                            CopyColumn(column_name_src="Incorrect Answer 3", column_name_dst="D"),
                            ShuffleColumns(columns=["A", "B", "C", "D"]),
                            # finds the answer choice that "Correct Answer" is mapped to, and stores it in "ground_truth"
                            ColumnMatchMap(new_col="ground_truth", key_col="Correct Answer", columns=["A", "B", "C", "D"]),
                        ]
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__),
                "../prompt_templates/gpqa_templates/basic.jinja",
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
                            CopyColumn(
                                column_name_src="model_output",
                                column_name_dst="raw_model_output",
                            ),
                            RegexTransform(
                                columns="model_output",
                                prompt_pattern=r"My answer is (\w)(?=\s|\W|$)",
                                case=True,
                            ),
                            ImputeNA(columns="model_output", value=""),
                        ]
                    ),
                },
            ),
            metric_config=MetricConfig(ExactMatch),
            aggregator_configs=[
                AggregatorConfig(CountAggregator, {"column_names": ["ExactMatch_result"], "normalize": True})
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )
        # # Configure the pipeline
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.evalreporting_comp,
            ],
            self.log_dir,
        )