import re
from dataclasses import dataclass

import pandas as pd

from .transform import DFTransformBase


@dataclass
class BA_Calendar_ExtractAnswer(DFTransformBase):
    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.model_answer_column] = df[self.model_output_column].apply(self.parse_output_answer)
        return df

    @staticmethod
    def parse_output_answer(response):
        """
        Parse the input string to extract answer of a given BA Calendar problems.
        Parameters:
            response (str): Input string containing answer X in the form of "Final Answer: X".
        Returns: 
            numerical_value (float): A numeric value representing the model's answer.
        """
        answer = ""

        # Try to find an answer in the "Final Answer: X" format
        print(response)
        match = re.search(r"(?i)(?<=Final Answer: ).*", response)
        print(match)
        if match:
            answer = match.group(0)

        return answer