"""Defines a transform that extracts code snippets from model outputs.

Reproduces the behavior of
https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/utils/extraction_utils.py
"""

import dataclasses
import pandas as pd
import re

from tqdm.auto import tqdm

from eureka_ml_insights.data_utils import transform


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
            lambda x: extract_last_code_block(
                x, self.closing_think_token)
        )
        return df


def extract_code_blocks(
        response: str | None, closing_think_token: str = "") -> list[str]:
    """Extracts all code snippets from a model response.

    Only considers text after the last `closing_think_token` if provided.

    Args:
        response: The model response as a string.
        closing_think_token: The token marking the end of the model's thought
            process (e.g., "</think>").

    Returns:
        A list of all extracted code snippets (possibly empty).
    """
    if not response:
        return []

    if closing_think_token and closing_think_token not in response:
        return []

    # Restrict to text after the last think token, if present
    response_to_consider = (
        response.rpartition(closing_think_token)[2].strip()
        if closing_think_token
        else response.strip()
    )

    if not response_to_consider:
        return []

    # Find all markdown-style code blocks, optionally with a language tag
    # Matches both ```python\n<code>\n``` and ```\n<code>\n```
    matches = re.findall(
        r"```(?:python)?\n(.*?)\n```", response_to_consider, re.DOTALL)

    # Strip whitespace around each code snippet
    return [m.strip() for m in matches]


def extract_last_code_block(
        response: str | None, closing_think_token: str = "") -> str:
    """Extracts the last code snippet from a model response.

    Args:
        response: The model response as a string.
        closing_think_token: The token marking the end of the model's thought
            process (e.g., "</think>").

    Returns:
        The last extracted code snippet, or an empty string if none found.
    """
    blocks = extract_code_blocks(response, closing_think_token)
    return blocks[-1] if blocks else ""
