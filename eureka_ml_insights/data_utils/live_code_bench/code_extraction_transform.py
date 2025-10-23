"""Defines a transform that extracts code snippets from model outputs.

Reproduces the behavior of lcb_runner/utils/extraction_utils.py in the
LiveCodeBench repository.
"""

import dataclasses
import pandas as pd

from tqdm.auto import tqdm
from typing import override

from eureka_ml_insights.data_utils import transform

from eureka_ml_insights.data_utils.live_code_bench import parsing

@dataclasses.dataclass
class CodeExtractionTransform(transform.DFTransformBase):
    """Extracts the code snippet from the model output.

    Attributes:
        model_output_column: The name of the column containing the model output.
        code_column: The name of the column to store the extracted code.
        closing_think_token: An optional token indicating the end of the code
            snippet. If provided, will only consider code after this token.
    """

    model_output_column: str
    code_column: str
    closing_think_token: str = ""

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts the code snippets from the model outputs.

        Args:
            df: The input DataFrame.

        Returns:
            The DataFrame with the extracted code snippets.
        """
        tqdm.pandas(
            desc=f"Extracting code from {self.model_output_column}"
        )

        df[self.code_column] = df[self.model_output_column].progress_apply(
            lambda x: parsing.extract_last_code_block(
                x, self.closing_think_token)
        )
        return df
