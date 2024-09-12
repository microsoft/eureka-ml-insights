import os

from eureka_ml_insights.configs.experiment_config import ExperimentConfig
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    HFDataReader,
    MMDataLoader,
    ColumnRename,
    DataReader,
    PrependStringTransform,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.spatial_utils import (
    LowerCaseNoPunctuationConvertNumbers,
)
from eureka_ml_insights.metrics import (
    CountAggregator,
    ObjectRecognitionMetric,
)

from ..config import (
    AggregatorConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    MetricConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from .common import LOCAL_DATA_PIPELINE

"""This file contains example user defined configuration classes for the visual prompting task.
In order to define a new configuration, a new class must be created that directly or indirectly
 inherits from ExperimentConfig and the configure_pipeline method should be implemented.
You can inherit from one of the existing user defined classes below and override the necessary
attributes to reduce the amount of code you need to write.

The user defined configuration classes are used to define your desired *pipeline* that can include
any number of *component*s. Find *component* options in the core module.

Pass the name of the class to the main.py script to run the pipeline.
"""


class VISUAL_PROMPTING_PAIRS_PIPELINE(ExperimentConfig):
    """
    This defines an ExperimentConfig pipeline for the visual prompting dataset, pairs condition.
    There is no model_config by default and the model config must be passed in via command lime.
    """

    def configure_pipeline(self, model_config, resume_from=None):
        # Configure the data processing component.
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "microsoft/IMAGE_UNDERSTANDING",
                    "split": "val",
                    "tasks": "visual_prompting_pairs",
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
                },
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
                            LowerCaseNoPunctuationConvertNumbers(columns=["model_output"]),
                        ]
                    ),
                },
            ),
            metric_config=MetricConfig(ObjectRecognitionMetric),
            aggregator_configs=[
                AggregatorConfig(
                    CountAggregator, {"column_names": ["ObjectRecognitionMetric_result"], "normalize": True}
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # Configure the pipeline
        return PipelineConfig([self.data_processing_comp, self.inference_comp, self.evalreporting_comp], self.log_dir)


class VISUAL_PROMPTING_SINGLE_PIPELINE(VISUAL_PROMPTING_PAIRS_PIPELINE):
    """This class extends VISUAL_PROMPTING_PAIRS_PIPELINE to use the single object condition."""

    def configure_pipeline(self, model_config, resume_from=None):
        config = super().configure_pipeline(model_config, resume_from)
        self.data_processing_comp.data_reader_config.init_args["tasks"] = (
            "visual_prompting_single"
        )
        return config


class VISUAL_PROMPTING_PAIRS_LOCAL_PIPELINE(LOCAL_DATA_PIPELINE, VISUAL_PROMPTING_PAIRS_PIPELINE):
    def configure_pipeline(self, model_config, resume_from=None):
        local_path = "/home/neel/data/spatial_understanding"
        return super().configure_pipeline(model_config, resume_from, local_path)


class VISUAL_PROMPTING_SINGLE_LOCAL_PIPELINE(LOCAL_DATA_PIPELINE, VISUAL_PROMPTING_SINGLE_PIPELINE):
    def configure_pipeline(self, model_config, resume_from=None):
        local_path = "/home/neel/data/spatial_understanding"
        return super().configure_pipeline(model_config, resume_from, local_path)
