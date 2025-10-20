"""Defines a transform that extracts code snippets from model outputs."""

import dataclasses
import pandas as pd

from eureka_ml_insights.data_utils import transform

from eureka_ml_insights.data_utils.live_code_bench import parsing

@dataclasses.dataclass
class CodeExtractionTransform(transform.DFTransformBase):
    """Extracts the code snippet from the model output."""

    model_output_column: str
    code_column: str
    closing_think_token: str = ""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.code_column] = df[self.model_output_column].apply(
            lambda x: parsing.extract_last_code_block(
                x, self.closing_think_token)
        )
        return df
