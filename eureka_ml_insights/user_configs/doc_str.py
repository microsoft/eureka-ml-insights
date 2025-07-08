"""
This module defines transformations for reading and writing file content
and an experiment pipeline configuration for generating docstrings.
"""

import os
from dataclasses import dataclass
from typing import Any

from eureka_ml_insights.configs import (
    DataSetConfig,
    ExperimentConfig,
    InferenceConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.core import DataProcessing, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    DataReader,
    DFTransformBase,
    MMDataLoader,
)


@dataclass
class FileReaderTransform(DFTransformBase):
    """
    A transformation that reads file content from a specified file path column.
    """

    file_path_column: str = "file_path"

    def transform(self, df):
        """
        Transform the DataFrame by reading content based on file paths.

        Args:
            df (pandas.DataFrame): The input DataFrame with the column specified
                by file_path_column.

        Returns:
            pandas.DataFrame: The DataFrame with a new column 'file_content'.
        """
        # Implement the logic to read files from the specified column
        df["file_content"] = df[self.file_path_column].apply(lambda x: open(x).read())
        return df


class FileWriterTransform(DFTransformBase):
    """
    A transformation that writes file content to a specified file path.
    """

    file_path_column: str = "file_path"
    file_content_column: str = "model_output"

    def transform(self, df):
        """
        Transforms the DataFrame by writing file content to disk.

        This method replaces certain path elements, creates output directories if needed,
        and writes the content from file_content_column to files specified by file_path_column.

        Args:
            df (pandas.DataFrame): The input DataFrame containing file path and content columns.

        Returns:
            pandas.DataFrame: The original DataFrame after writing the content to disk.
        """
        # replace "/home/sayouse/git/eureka-ml-insights/" with "/home/sayouse/git/eureka-ml-insights-doc/" in the file paths
        output_file_path = df[self.file_path_column].apply(
            lambda x: x.replace("/home/sayouse/git/eureka-ml-insights/", "/home/sayouse/git/eureka-ml-insights-doc/")
        )
        # if the output file path does not exist, create the directory
        output_file_path.apply(lambda x: os.makedirs(os.path.dirname(x), exist_ok=True))
        # Implement the logic to write files to the specified column
        for index, row in df.iterrows():
            with open(row[self.file_path_column], "w") as f:
                f.write(row[self.file_content_column])
        return df


class DOCSTR_PIPELINE(ExperimentConfig):
    """
    An experiment configuration for a docstring generation pipeline.
    """

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        """
        Configure the pipeline components for docstring generation.

        Args:
            model_config (ModelConfig): Configuration for the model.
            resume_from (str, optional): Path to a checkpoint to resume from. Defaults to None.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            PipelineConfig: The configured pipeline consisting of data processing,
            inference, and post-processing components.
        """
        # Configure the data processing component.
        data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            prompt_template_path=os.path.join(os.path.dirname(__file__), "../prompt_templates/doc_str.jinja"),
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": "/home/sayouse/git/eureka-ml-insights/eureka_ml_insights/python_files_m.csv",
                    "format": ".csv",
                    "header": 0,
                    "index_col": None,
                    "transform": FileReaderTransform(),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )
        inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {"path": os.path.join(data_processing_comp.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(self.log_dir, "inference_output"),
            resume_from=resume_from,
            max_concurrent=5,
        )
        post_process_comp = PromptProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(inference_comp.output_dir, "inference_result.jsonl"),
                    "transform": FileWriterTransform(),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_post_processing_output"),
        )
        return PipelineConfig([data_processing_comp, inference_comp, post_process_comp])
