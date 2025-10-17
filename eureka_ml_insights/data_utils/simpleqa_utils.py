import ast
import json
import math
import re
from dataclasses import dataclass

import pandas as pd

from .transform import DFTransformBase

@dataclass
class SimpleQA_MetadataExplode(DFTransformBase):
    metadata_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.metadata_column] = df[self.metadata_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        self.explode_metadata(df)
        return df
    
    def explode_metadata(self, df):
        # TODO this would break if the first row does not have all the metrics, e.g. due invalid inference results
        for col_name in df[self.metadata_column][0].keys():
            df[col_name] = df.apply(
                lambda row: (
                    row[self.metadata_column].get(col_name, None)
                    if isinstance(row[self.metadata_column], dict)
                    else None
                ),
                axis=1,
            )
        df.drop(self.metadata_column, axis=1, inplace=True)
        return df