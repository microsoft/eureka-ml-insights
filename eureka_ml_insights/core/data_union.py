"""Module for data union functionality.

This module provides a DataUnion class which concatenates two datasets using pandas concat functionality.
"""

from typing import List, Optional

import pandas as pd

from .data_processing import DataProcessing


class DataUnion(DataProcessing):
    """Concatenates two datasets using pandas concat functionality."""

    def __init__(
        self,
        data_reader_config,
        output_dir: str,
        other_data_reader_config,
        output_data_columns: Optional[List[str]] = None,
        dedupe_cols: Optional[List[str]] = None,
    ) -> None:
        """Initializes DataUnion.

        Args:
            data_reader_config: DataReaderConfig object for reading the primary dataset.
            output_dir (str): Directory to save the output files of this component.
            other_data_reader_config: DataReaderConfig object for reading the secondary dataset.
            output_data_columns (Optional[List[str]]): List of columns (subset of input columns) to keep
                in the transformed data output file.
            dedupe_cols (Optional[List[str]]): List of columns to deduplicate the concatenated DataFrame.
        """
        super().__init__(data_reader_config, output_dir, output_data_columns)
        self.other_data_reader = other_data_reader_config.class_name(**other_data_reader_config.init_args)
        self.dedupe_cols = dedupe_cols

    @classmethod
    def from_config(cls, config):
        """Creates a DataUnion instance from a configuration object.

        Args:
            config: Configuration object containing initialization parameters.

        Returns:
            DataUnion: The newly created DataUnion instance.
        """
        return cls(
            config.data_reader_config,
            config.output_dir,
            config.other_data_reader_config,
            config.output_data_columns,
            config.dedupe_cols,
        )

    def run(self):
        """Concatenates two DataFrames, optionally deduplicates them, and writes the result.

        This method:
         - Loads two datasets using the configured data readers.
         - Concatenates them with pandas.concat.
         - Optionally drops duplicates based on dedupe_cols.
         - Filters columns to output_data_columns if provided.
         - Writes the final DataFrame to the specified output directory.
        """
        df = self.data_reader.load_dataset()
        other_df = self.other_data_reader.load_dataset()
        if len(df.columns) > 0 and len(other_df.columns) > 0:
            concat_df = pd.concat([df, other_df], axis=0)
        # handling corner cases when one of the data frames is empty
        elif len(df.columns) == 0:
            concat_df = other_df
        elif len(other_df.columns) == 0:
            concat_df = df
        self.output_data_columns = [col for col in self.output_data_columns if col in concat_df.columns]
        if len(concat_df.columns) > 0:
            concat_df = self.get_desired_columns(concat_df)
        if self.dedupe_cols:
            concat_df = concat_df.drop_duplicates(subset=self.dedupe_cols)
        self.write_output(concat_df)
