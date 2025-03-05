from typing import List, Optional

import pandas as pd

from .data_processing import DataProcessing


class DataUnion(DataProcessing):
    """This component is used to join two datasets using pandas merge functionality."""

    def __init__(
        self,
        data_reader_config,
        output_dir: str,
        other_data_reader_config,
        output_data_columns: Optional[List[str]] = None,
    ) -> None:
        """
        args:
            data_reader_config: DataReaderConfig
            output_dir: str directory to save the output files of this component.
            other_data_reader_config: DataReaderConfig
            output_data_columns: Optional[List[str]] list of columns (subset of input columns)
                                      to keep in the transformed data output file.
        """
        super().__init__(data_reader_config, output_dir, output_data_columns)
        self.other_data_reader = other_data_reader_config.class_name(**other_data_reader_config.init_args)

    @classmethod
    def from_config(cls, config):
        return cls(
            config.data_reader_config,
            config.output_dir,
            config.other_data_reader_config,
            config.output_data_columns,
        )

    def run(self):
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
        concat_df = concat_df.drop_duplicates()
        self.write_output(concat_df)
