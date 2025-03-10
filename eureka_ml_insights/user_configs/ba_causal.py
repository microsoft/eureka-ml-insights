import os
from typing import Any
import math

from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    ColumnRename,
    DataReader,
    HFDataReader,
    MMDataLoader,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.transform import AddColumn, CopyColumn, RegexTransform, RunPythonTransform, SamplerTransform
from eureka_ml_insights.metrics import CountAggregator, GeoMCQMetric

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
from eureka_ml_insights.configs import ExperimentConfig
from eureka_ml_insights.metrics.metrics_base import CaseInsensitiveMatch, ExactMatch

"""This file contains user defined configuration classes for the geometric reasoning task on geometer dataset.
"""

class BA_Causal_PIPELINE(ExperimentConfig):
    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        # Configure the data processing component.
        data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            prompt_template_path=os.path.join(
                # os.path.dirname(__file__), "../prompt_templates/ba_causal_templates/ba_causal_prompt.jinja"
                # os.path.dirname(__file__), "../prompt_templates/ba_causal_templates/ba_causal_descriptive.jinja"
                os.path.dirname(__file__), "../prompt_templates/ba_causal_templates/ba_causal_step.jinja"
            ),
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join("/mnt/c/Users/vidhishab/OneDrive - Microsoft/Research/code/local_benchmark_data/Natasha_benchmarks/datasets/datasets/ba-causal/sorted_instances_500.jsonl"),
                    "transform": SequenceTransform(
                        [
                            AddColumn("ground_truth"),
                            AddColumn("images"),
                            RunPythonTransform("df['ground_truth'] = df['metadata'].apply(lambda x: x['answer'])"),
                            RunPythonTransform("df['images'] = df['metadata'].apply(lambda x: x['inputs']+x['options'])"),
                            # RunPythonTransform("df['images'] = df['metadata'].apply(lambda x: x['inputs_large']+x['options_large'])"),
                            # RunPythonTransform("df['images'] = df['metadata'].apply(lambda x: x['single_image'])"),
                            # SamplerTransform(sample_count=10, random_seed=99)
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        # Configure the inference component
        inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {
                    "path": os.path.join(data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "image_column_names": ["images"]
                },
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
            max_concurrent=5
        )

        # # Configure the evaluation and reporting component.
        evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform([
                        CopyColumn("model_output", "raw_output"),
                        RegexTransform(
                                columns="model_output",
                                prompt_pattern=r"Answer: (\w)(?=\s|\W|$)",
                                ignore_case=False,
                            ),
                        RunPythonTransform("df['complexity_bucket'] = df['metadata'].apply(lambda x: round(math.floor(x['complexity'] / 0.01) * 0.1, 4))", global_imports=["math"]),
                    ])
                },
            ),
            metric_config=MetricConfig(CaseInsensitiveMatch),
            aggregator_configs=[
                AggregatorConfig(CountAggregator, {"column_names": ["CaseInsensitiveMatch_result"], "normalize": True}),
                AggregatorConfig(
                    CountAggregator,
                    {"column_names": ["CaseInsensitiveMatch_result"], "group_by": "complexity_bucket", "normalize": True},
                ),
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
