import re
from dataclasses import dataclass

import pandas as pd

from .transform import DFTransformBase

"""This module provides classes and functions that extract numeric answers from textual model 
output for AIME questions."""


@dataclass
class AIMEExtractAnswer(DFTransformBase):
    """A data transformation class that extracts an AIME answer from the model's output.

    Attributes:
        model_output_column (str): The name of the column in the DataFrame containing the model output.
        model_answer_column (str): The name of the column in the DataFrame where the extracted answer
            will be stored.
    """

    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the given DataFrame by extracting the AIME answer from the model's output column.

        Args:
            df (pd.DataFrame): The input DataFrame to be transformed.

        Returns:
            pd.DataFrame: The transformed DataFrame with the extracted answers in the specified column.
        """
        df[self.model_answer_column] = df[self.model_output_column].apply(self.parse_output_answer)
        return df

    @staticmethod
    def parse_output_answer(response):
        """Parses the input string to extract the answer of a given AIME question.

        Args:
            response (str): Input string containing the answer in the form of "Final Answer: X".

        Returns:
            float: A numeric value representing the model's answer.
        """
        numerical_value = None

        # Try to find an answer in the "Final Answer: X" format
        match = re.search(r"Final Answer:\s*([\$]?-?[\d,]+(?:\.\d+)?%?)", response)
        # If not found, try to find an answer in the "Final Answer: [X]" format
        if not match:
            match = re.search(r"Final Answer:\s*\[([\$]?-?[\d,]+(?:\.\d+)?%?)\]", response)
        if match:
            answer_str = match.group(1)
            # Remove $ and commas, handle percentages for numerical comparison
            answer_str = answer_str.replace("$", "").replace(",", "")
            if answer_str.endswith("%"):
                numerical_value = float(answer_str[:-1]) / 100  # Convert percentage to decimal
            else:
                try:
                    numerical_value = float(answer_str)
                except ValueError:
                    numerical_value = None

        return numerical_value
