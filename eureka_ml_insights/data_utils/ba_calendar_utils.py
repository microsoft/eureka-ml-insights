"""
This module provides the BA_Calendar_ExtractAnswer class, which transforms a DataFrame
to extract answers from a model output column for BA Calendar problems.
"""

import re
from dataclasses import dataclass

import pandas as pd

from .transform import DFTransformBase


@dataclass
class BA_Calendar_ExtractAnswer(DFTransformBase):
    """Extracts answers from a model output column in a DataFrame.

    This class extends DFTransformBase, providing a transformation method to parse
    and extract BA Calendar problem answers from a specified output column.

    Attributes:
        model_output_column (str): The name of the column containing the model output.
        model_answer_column (str): The name of the column that will store the parsed answers.
    """

    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the given DataFrame by parsing the model output column to extract answers.

        Args:
            df (pd.DataFrame): The input DataFrame containing the model output column.

        Returns:
            pd.DataFrame: The transformed DataFrame with a new column containing the parsed answers.
        """
        df[self.model_answer_column] = df[self.model_output_column].apply(self.parse_output_answer)
        return df

    @staticmethod
    def parse_output_answer(response):
        """Parses the input string to extract an answer for BA Calendar problems.

        Args:
            response (str): The input string containing the answer in the form of "Final Answer: X".

        Returns:
            str: A string representing the extracted answer. If no valid answer is found,
            returns an empty string or "No common time slot available" if applicable.
        """
        answer = ""

        if response is None:
            return ""
        
        response = response.replace("**", "").replace("\n", "")

        match = re.findall(r"Final Answer:\s*(\w+ \d{2}:\d{2}-\d{2}:\d{2})", response)
        
        if match:
            answer = match[len(match) - 1]
        elif "No common time slot available".lower() in response.lower():
            answer = "No common time slot available"

        return answer