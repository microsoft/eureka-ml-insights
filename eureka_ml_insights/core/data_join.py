"""Module for joining datasets using pandas merge functionality.

This module defines the DataJoin class, which inherits from DataProcessing, and provides methods
to merge two pandas DataFrames with flexible join types. The module also includes a factory
class method for construction from a configuration object.
"""

from typing import List, Optional

import pandas as pd

from .data_processing import DataProcessing


class DataJoin(DataProcessing):
    """Joins two datasets using pandas merge functionality."""

    def __init__(
        self,
        data_reader_config,
        output_dir: str,
        other_data_reader_config,
        pandas_merge_args: dict,
        output_data_columns: Optional[List[str]] = None,
    ) -> None:
        """Initializes the DataJoin component.

        Args:
            data_reader_config: Configuration object for reading the first dataset.
            output_dir (str): Directory to save the output files of this component.
            other_data_reader_config: Configuration object for reading the second dataset.
            pandas_merge_args (dict): Arguments passed to pandas merge function.
            output_data_columns (List[str], optional): List of columns (subset of input columns)
                to keep in the transformed data output file.
        """
        super().__init__(data_reader_config, output_dir, output_data_columns)
        self.other_data_reader = other_data_reader_config.class_name(**other_data_reader_config.init_args)
        self.pandas_merge_args = pandas_merge_args
        allowed_join_types = {"inner", "outer", "left", "right", "cross"}
        join_type = self.pandas_merge_args["how"]
        if join_type.lower() not in allowed_join_types:
            raise ValueError(f"Invalid join type '{join_type}'. Expected one of: {', '.join(allowed_join_types)}.")

    @classmethod
    def from_config(cls, config):
        """Creates a DataJoin instance from a configuration object.

        Args:
            config: A configuration object containing all required initialization attributes.

        Returns:
            DataJoin: An instance of the DataJoin class.
        """
        return cls(
            config.data_reader_config,
            config.output_dir,
            config.other_data_reader_config,
            config.pandas_merge_args,
            config.output_data_columns,
        )

    def run(self):
        """Executes the data join operation and writes the result to the output directory."""
        df = self.data_reader.load_dataset()
        other_df = self.other_data_reader.load_dataset()
        if len(df.columns) > 0 and len(other_df.columns) > 0:
            joined_df = pd.merge(df, other_df, **self.pandas_merge_args)
        # handling corner cases when one of the data frames is empty
        elif len(df.columns) == 0:
            if self.pandas_merge_args["how"] == "inner":
                joined_df = df
            elif self.pandas_merge_args["how"] == "outer":
                joined_df = other_df
            elif self.pandas_merge_args["how"] == "left":
                joined_df = df
            elif self.pandas_merge_args["how"] == "right":
                joined_df = other_df
            elif self.pandas_merge_args["how"] == "cross":
                joined_df = df
        elif len(other_df.columns) == 0:
            if self.pandas_merge_args["how"] == "inner":
                joined_df = other_df
            elif self.pandas_merge_args["how"] == "outer":
                joined_df = df
            elif self.pandas_merge_args["how"] == "left":
                joined_df = df
            elif self.pandas_merge_args["how"] == "right":
                joined_df = other_df
            elif self.pandas_merge_args["how"] == "cross":
                joined_df = other_df
        if len(joined_df.columns) > 0:
            joined_df = self.get_desired_columns(joined_df)
        self.write_output(joined_df)
