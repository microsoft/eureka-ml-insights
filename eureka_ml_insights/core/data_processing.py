import json
import logging
import os
from hashlib import md5
from typing import List, Optional

from eureka_ml_insights.configs.config import DataProcessingConfig, DataSetConfig
from eureka_ml_insights.data_utils import NumpyEncoder

"""
This module defines a data processing pipeline component and integrates reserved name handling for data outputs.
"""

from .pipeline import Component
from .reserved_names import (
    INFERENCE_RESERVED_NAMES,
    PROMPT_PROC_RESERVED_NAMES,
)

class DataProcessing(Component):
    """Implements data reading, transformation, and output writing for a pipeline component."""

    @classmethod
    def from_config(cls, config: DataProcessingConfig):
        """Create a DataProcessing instance from a configuration object.

        Args:
            config: A DataProcessingConfig object with data_reader_config,
                output_dir, and output_data_columns.

        Returns:
            DataProcessing: An instance of DataProcessing.
        """
        return cls(
            config.data_reader_config,
            config.output_dir,
            config.output_data_columns,
        )

    def __init__(
        self,
        data_reader_config: DataSetConfig,
        output_dir: str,
        output_data_columns: Optional[List[str]] = None,
    ) -> None:
        """Initialize the DataProcessing component.

        Args:
            data_reader_config (DataSetConfig): The configuration for reading the input data.
            output_dir (str): Directory to save the output files of this component.
            output_data_columns (Optional[List[str]], optional): A list of columns (subset of input columns)
                to keep in the transformed data output file. The columns reserved for the Eureka framework
                will automatically be added to the output_data_columns if not provided.
        """
        super().__init__(output_dir)
        self.data_reader = data_reader_config.class_name(**data_reader_config.init_args)
        self.output_data_columns = output_data_columns

    def write_output(self, df):
        """Write the transformed DataFrame to a JSONL file in the component's output directory.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data to write.
        """
        logging.info(f"About to save transformed_data_file with columns: {df.columns}.")
        transformed_data_file = os.path.join(self.output_dir, "transformed_data.jsonl")

        with open(transformed_data_file, "w", encoding="utf-8") as writer:
            for _, row in df.iterrows():
                content = row.to_dict()
                writer.write(json.dumps(content, ensure_ascii=False, separators=(",", ":"), cls=NumpyEncoder) + "\n")

    def get_desired_columns(self, df):
        """Get the desired columns from the DataFrame, including necessary reserved columns.

        Args:
            df (pandas.DataFrame): The DataFrame from which columns will be selected.

        Returns:
            pandas.DataFrame: The DataFrame containing only the desired columns.
        """
        if self.output_data_columns is None:
            self.output_data_columns = df.columns
        self.output_data_columns = list(self.output_data_columns)
        # if the data was multiplied, keep the columns that are needed to identify datapoint and replicates
        # (just in case the user forgot to specify these columns in output_data_columns)
        cols_to_keep = set(INFERENCE_RESERVED_NAMES + PROMPT_PROC_RESERVED_NAMES)
        self.output_data_columns.extend([col for col in cols_to_keep if col in df.columns])
        self.output_data_columns = list(set(self.output_data_columns))
        return df[self.output_data_columns]

    def run(self) -> None:
        """Run the data processing steps.

        Loads the dataset, applies transformations (if any),
        selects desired columns, and writes the output as a JSONL file.
        """
        input_df = self.data_reader.load_dataset()
        logging.info(f"input has: {len(input_df)} rows, and the columns are: {input_df.columns}.")
        # if input_df is not empty, select the desired columns
        if not input_df.empty:
            input_df = self.get_desired_columns(input_df)
        self.write_output(input_df)
