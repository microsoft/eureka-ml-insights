import os
from typing import Any

from eureka_ml_insights.configs.experiment_config import ExperimentConfig
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    ColumnRename,
    CopyColumn,
    DataReader,
    ExtractBoundingBox,
    SamplerTransform,
    MMDataLoader,
    SequenceTransform,
)
from eureka_ml_insights.metrics import CountAggregator, BiLevelCountAggregator, BboxMetric

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


class SCREENSPOT_NORMALIZED_PIPELINE(ExperimentConfig):
    """
    This defines an ExperimentConfig pipeline for the SCREENSPOT dataset using normalized coordinates.
    There is no model_config by default and the model config must be passed in via command line.
    """

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any] ) -> PipelineConfig:
    
        self.data_processing_comp = PromptProcessingConfig(
        component_type=PromptProcessing,
        data_reader_config=DataSetConfig(
            DataReader,
            {
                "path": "/mnt/phimmwestus3_datasets/EVAL/ScreenSpot/screenspot_all.jsonl",                
                "transform": SequenceTransform(
                    [                 
                        ColumnRename(name_mapping={"img_filename": "image"}),
                    ]
                ),
            },          
        ),
        prompt_template_path=os.path.join(
            os.path.dirname(__file__), "../prompt_templates/screenspot_templates/normalized.jinja"
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
                {
                    "path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "mm_data_path_prefix": "/mnt/phimmwestus3_datasets/EVAL/ScreenSpot/images",
                 },
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
            max_concurrent=20,
        )

        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={"model_output": "model_output_raw"}),
                            ExtractBoundingBox(
                                answer_column_name="model_output_raw",
                                extracted_answer_column_name="model_output",
                            ),
                        ],
                    ),
                },
            ),
            metric_config=MetricConfig(BboxMetric, {"normalized": True}),
            aggregator_configs=[
                AggregatorConfig(CountAggregator, {"column_names": ["BboxMetric_result"], "normalize": True}),
                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": [
                            "BboxMetric_result",
                        ],
                        "first_groupby": "group",
                        "second_groupby": "data_type",
                        "normalize": True,
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # Configure the pipeline
        return PipelineConfig([self.data_processing_comp, self.inference_comp, self.evalreporting_comp], self.log_dir)

class SCREENSPOT_UNNORMALIZED_PIPELINE(SCREENSPOT_NORMALIZED_PIPELINE):
    """
    This defines an ExperimentConfig pipeline for the SCREENSPOT dataset using normalized coordinates.
    There is no model_config by default and the model config must be passed in via command line.
    """

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any] ) -> PipelineConfig:
        config = super().configure_pipeline(model_config, resume_from)
        self.data_processing_comp.prompt_template_path = os.path.join(os.path.dirname(__file__), "../prompt_templates/screenspot_templates/unnormalized.jinja"            )
        self.evalreporting_comp.metric_config = MetricConfig(BboxMetric, {"normalized": False})
        return config
    
class SCREENSPOT_PRO_NORMALIZED_PIPELINE(SCREENSPOT_NORMALIZED_PIPELINE):
    """
    This defines an ExperimentConfig pipeline for the SCREENSPOT_PRO dataset using normalized coordinates.
    There is no model_config by default and the model config must be passed in via command line.
    """

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any] ) -> PipelineConfig:
        config = super().configure_pipeline(model_config, resume_from)
        self.data_processing_comp.data_reader_config.init_args["path"] = "/mnt/phimmwestus3_datasets/EVAL/ScreenSpot-Pro/screenspot_pro_all.jsonl"
        self.inference_comp.data_loader_config.init_args["mm_data_path_prefix"] = "/mnt/phimmwestus3_datasets/EVAL/ScreenSpot-Pro/images"
        return config
    
class SCREENSPOT_PRO_UNNORMALIZED_PIPELINE(SCREENSPOT_UNNORMALIZED_PIPELINE):
    """
    This defines an ExperimentConfig pipeline for the SCREENSPOT_PRO dataset using unnormalized coordinates.
    There is no model_config by default and the model config must be passed in via command line.
    """

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any] ) -> PipelineConfig:
        config = super().configure_pipeline(model_config, resume_from)
        self.data_processing_comp.data_reader_config.init_args["path"] = "/mnt/phimmwestus3_datasets/EVAL/ScreenSpot-Pro/screenspot_pro_all.jsonl"
        self.inference_comp.data_loader_config.init_args["mm_data_path_prefix"] = "/mnt/phimmwestus3_datasets/EVAL/ScreenSpot-Pro/images"
        return config        