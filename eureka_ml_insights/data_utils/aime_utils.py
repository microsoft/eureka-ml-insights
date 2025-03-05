import re
from dataclasses import dataclass

import pandas as pd

from .transform import DFTransformBase


@dataclass
class AIMEExtractAnswer(DFTransformBase):
    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.model_answer_column] = df[self.model_output_column].apply(self.parse_output_answer)
        return df

    @staticmethod
    def parse_output_answer(response):
        """
        Parse the input string to extract answer of a given AIME question.
        Parameters:
            response (str): Input string containing answer X in the form of "Final Answer: X".
        Returns: 
            numerical_value (float): A numeric value representing the model's answer.
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
                except ValueError as e:
                    numerical_value = None

        return numerical_value