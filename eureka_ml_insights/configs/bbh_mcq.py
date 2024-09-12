import os

from eureka_ml_insights.configs import (
    DataSetConfig,
    InferenceConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.core import Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    ColumnRename,
    HFDataReader,
    MMDataLoader,
    SequenceTransform,
)
from eureka_ml_insights.models import HuggingFaceLM, OpenAIModelsAzure

from .experiment_config import ExperimentConfig


class BBH_MCQ_ORCA_PIPELINE(ExperimentConfig):
    """This method is used to define an eval pipeline with for BBH benchmark specifically for MCQ and MCQ like tasks using the Orca model"""

    def configure_pipeline(self):
        data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "lukaemon/bbh",
                    "split": "test",
                    "tasks": ["boolean_expressions", "causal_judgement"],
                    "transform": SequenceTransform(
                        [ColumnRename(name_mapping={"input": "prompt", "target": "ground_truth"})]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        # Inference component

        inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=ModelConfig(
                HuggingFaceLM,
                {
                    "model": "microsoft/Orca-2-7b",
                },
            ),
            data_loader_config=DataSetConfig(
                MMDataLoader, {"path": os.path.join(self.log_dir, "data_processing_output/transformed_data.jsonl")}
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
        )

        # Configure the pipeline
        return PipelineConfig([data_processing_comp, inference_comp], self.log_dir)


class BBH_MCQ_OpenAI_PIPELINE(BBH_MCQ_ORCA_PIPELINE):
    """This method is used to define an eval pipeline with for BBH benchmark specifically for MCQ and MCQ like tasks and uses an OpenAI model."""

    def configure_pipeline(self):
        config = super().configure_pipeline()
        # Inference component
        # config for OpenAI endpoint (this is just a PLACEHOLDER for now)
        openai_config = {
            "azure_endpoint": "https://gcraoai8sw1.openai.azure.com/",
            "api_key": "",
            "model_name": "gpt-35-turbo",
        }

        inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=ModelConfig(
                OpenAIModelsAzure,
                {
                    "config": openai_config,
                },
            ),
            data_loader_config=DataSetConfig(
                MMDataLoader, {"path": os.path.join(self.log_dir, "data_processing_output/transformed_data.jsonl")}
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
        )

        config.component_configs[1] = inference_comp
        # Configure the pipeline
        return config
