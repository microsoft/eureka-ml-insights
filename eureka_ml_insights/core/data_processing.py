import base64
import json
import logging
import os
from hashlib import md5
from typing import List, Optional

import numpy as np

from .pipeline import Component


def compute_hash(val: str) -> str:
    """
    Hashes the provided value using MD5.
    """
    return md5(val.encode("utf-8")).hexdigest()


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return base64.b64encode(obj).decode("ascii")
        return json.JSONEncoder.default(self, obj)


class DataProcessing(Component):
    @classmethod
    def from_config(cls, config):
        return cls(
            config.data_reader_config,
            config.output_dir,
            config.output_data_columns,
        )

    def __init__(
        self,
        data_reader_config,
        output_dir: str,
        output_data_columns: Optional[List[str]] = None,
    ) -> None:
        """
        args:
            data_reader_config: DataReaderConfig
            output_dir: str directory to save the output files of this component.
            output_data_columns: Optional[List[str]] list of columns (subset of input columns)
                                      to keep in the transformed data output file.
        """
        super().__init__(output_dir)
        self.data_reader = data_reader_config.class_name(**data_reader_config.init_args)
        self.output_data_columns = output_data_columns

    def write_output(self, df):
        logging.info(f"About to save transformed_data_file with columns: {df.columns}.")
        transformed_data_file = os.path.join(self.output_dir, "transformed_data.jsonl")

        with open(transformed_data_file, "w", encoding="utf-8") as writer:
            for _, row in df.iterrows():
                content = row.to_dict()
                writer.write(json.dumps(content, ensure_ascii=False, separators=(",", ":"), cls=NumpyEncoder) + "\n")

    def get_desired_columns(self, df):
        if self.output_data_columns is None:
            self.output_data_columns = df.columns
        self.output_data_columns = list(self.output_data_columns)
        # if the data was multiplied, keep the columns that are needed to identify datapoint and replicates
        # (just in case the user forgot to specify these columns in output_data_columns)
        cols_to_keep = ["data_point_id", "data_repeat_id"]
        self.output_data_columns.extend([col for col in cols_to_keep if col in df.columns])
        self.output_data_columns = list(set(self.output_data_columns))
        return df[self.output_data_columns]

    def run(self) -> None:
        # data reader loads data into a pandas dataframe and applies any transformations
        input_df = self.data_reader.load_dataset()
        logging.info(f"input has: {len(input_df)} rows, and the columns are: {input_df.columns}.")
        input_df = self.get_desired_columns(input_df)
        self.write_output(input_df)
