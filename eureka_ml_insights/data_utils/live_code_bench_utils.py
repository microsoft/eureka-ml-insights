"""Defines some transformation utilities for the LiveCodeBench benchmark."""

import dataclasses
import re

from eureka_ml_insights.data_utils import transform


@dataclasses.dataclass
class CodeExtractionTransform(transform.DFTransformBase):
    """Extracts the code snippet from the model output."""

    model_output_column: str
    code_column: str

    def transform(self, df):
        df[self.code_column] = (
            df[self.model_output_column]
            .apply(self.extract_code))
        return df

    @staticmethod
    def extract_code(response):
        if response is None:
            return None

        if not isinstance(response, str):
            return None

        # Try to find a code snippet in markdown format
        match = re.search(r"```(?:python)?\n(.*?)\n```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # If not found, return the whole response as a fallback
        return response.strip()