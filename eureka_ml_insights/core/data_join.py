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
        allowed_join_types = {"inner", "outer", "left", "right", "cross"}
        join_type = self.pandas_merge_args["how"]
        if join_type.lower() not in allowed_join_types:
            raise ValueError(
                f"Invalid join type '{join_type}'. Expected one of: {', '.join(allowed_join_types)}."
            )
        
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
        if len(df.columns) > 0 and len(other_df.columns) > 0:
            joined_df = pd.merge(df, other_df, **self.pandas_merge_args)
        # handling corner cases when one of the data frames is empty
        elif len(df.columns) == 0:
            if (self.pandas_merge_args["how"] == 'inner'):
                joined_df = df
            elif (self.pandas_merge_args["how"] == 'outer'):
                joined_df = other_df
            elif (self.pandas_merge_args["how"] == 'left'):
                joined_df = df
            elif (self.pandas_merge_args["how"] == 'right'):
                joined_df = other_df
            elif (self.pandas_merge_args["how"] == 'cross'):
                joined_df = df
        elif len(other_df.columns) == 0:
            if (self.pandas_merge_args["how"] == 'inner'):
                joined_df = other_df
            elif (self.pandas_merge_args["how"] == 'outer'):
                joined_df = df
            elif (self.pandas_merge_args["how"] == 'left'):
                joined_df = df
            elif (self.pandas_merge_args["how"] == 'right'):
                joined_df = other_df
            elif (self.pandas_merge_args["how"] == 'cross'):
                joined_df = other_df
        if (len(joined_df.columns) > 0):
            joined_df = self.get_desired_columns(joined_df)
        self.write_output(joined_df)