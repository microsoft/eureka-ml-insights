from typing import List, Optional

import pandas as pd

from .data_processing import DataProcessing


class DataJoin(DataProcessing):
    """This component is used to join two datasets using pandas merge functionality."""

    def __init__(
        self,
        data_reader_config,
        output_dir: str,
        other_data_reader_config,
        pandas_merge_args: dict,
        output_data_columns: Optional[List[str]] = None,
    ) -> None:
        """
        args:
            data_reader_config: DataReaderConfig
            output_dir: str directory to save the output files of this component.
            other_data_reader_config: DataReaderConfig
            pandas_merge_args: dict arguments to be passed to pandas merge function.
            output_data_columns: Optional[List[str]] list of columns (subset of input columns)
                                      to keep in the transformed data output file.
        """
        super().__init__(data_reader_config, output_dir, output_data_columns)
        self.other_data_reader = other_data_reader_config.class_name(**other_data_reader_config.init_args)
        self.pandas_merge_args = pandas_merge_args

    @classmethod
    def from_config(cls, config):
        return cls(
            config.data_reader_config,
            config.output_dir,
            config.other_data_reader_config,
            config.pandas_merge_args,
            config.output_data_columns,
        )

    def run(self):
        df = self.data_reader.load_dataset()
        other_df = self.other_data_reader.load_dataset()
        joined_df = pd.merge(df, other_df, **self.pandas_merge_args)
        joined_df = self.get_desired_columns(joined_df)
        self.write_output(joined_df)
