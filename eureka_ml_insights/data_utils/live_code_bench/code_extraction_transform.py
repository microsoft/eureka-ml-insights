"""Defines a transform that extracts code snippets from model outputs."""

import dataclasses

from eureka_ml_insights.data_utils import transform

from eureka_ml_insights.data_utils.live_code_bench import parsing

@dataclasses.dataclass
class CodeExtractionTransform(transform.DFTransformBase):
    """Extracts the code snippet from the model output."""

    model_output_column: str
    code_column: str

    def transform(self, df):
        df[self.code_column] = (
            df[self.model_output_column].apply(parsing.extract_code))
        return df
