import os

from eureka_ml_insights.configs.experiment_config import ExperimentConfig
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing, DataProcessing

from eureka_ml_insights.data_utils import (
    HFDataReader,    
    MMDataLoader,
    CopyColumn,
    ExtractUsageTransform,
    ReplaceStringsTransform,
    RunPythonTransform,
    MajorityVoteTransform,
    ColumnRename,
    DataReader,
    ExtractQuestionOptions,
    ExtractAnswerSpatialMapAndMaze,
    MultiplyTransform,
    SamplerTransform,
    SequenceTransform,
    ExtractConversations,
)
from eureka_ml_insights.metrics import SubstringExistsMatch, BiLevelAggregator, BiLevelCountAggregator, CountAggregator

from eureka_ml_insights.configs import (
    AggregatorConfig,
    DataProcessingConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)

"""This file contains example user defined configuration classes for the maze task.
In order to define a new configuration, a new class must be created that directly or indirectly
 inherits from UserDefinedConfig and the user_init method should be implemented.
You can inherit from one of the existing user defined classes below and override the necessary
attributes to reduce the amount of code you need to write.

The user defined configuration classes are used to define your desired *pipeline* that can include
any number of *component*s. Find *component* options in the core module.

Pass the name of the class to the main.py script to run the pipeline.
"""


class VISION_TRAINING_DATA(ExperimentConfig):
    """This method is used to define an eval pipeline with inference and metric report components,
    on any json formatted vision training dataset."""

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None) -> PipelineConfig:
        # Configure the data processing component.
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": "/mnt/e/Source/eureka/logs/bunny_llava_1.4m.json",
                    "transform": SequenceTransform(
                        [
                            SamplerTransform(random_seed=42, sample_count=100000),
                            ExtractConversations(),
                            #MultiplyTransform(n_repeats=5),
                        ],
                    ),                    
                },
            ),
            output_data_columns=["id", "prompt", "ground_truth"],            
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__),
                "../prompt_templates/vision_training_data_templates/no_image.jinja",
            ),            
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
            max_concurrent=100,
        )

        # Configure the pipeline
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
            ],
            self.log_dir,
        )