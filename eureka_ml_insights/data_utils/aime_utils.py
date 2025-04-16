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
        Parse the input string to extract the final answer of a given AIME question.
        Parameters:
            response (str): Input string containing answer(s) in the form of "Final Answer: X".
        Returns: 
            numerical_value (float or None): A numeric value representing the model's answer.
        """
        numerical_value = None

        # Find all matches in the format "Final Answer: X"
        matches = re.findall(r"Final Answer:\s*\[?([\$]?-?[\d,]+(?:\.\d+)?%?)\]?", response)
        
        if matches:
            # Take the last match as the final answer
            answer_str = matches[-1]
            answer_str = answer_str.replace("$", "").replace(",", "")
            if answer_str.endswith("%"):
                try:
                    numerical_value = float(answer_str[:-1]) / 100
                except ValueError:
                    numerical_value = None
            else:
                try:
                    numerical_value = float(answer_str)
                except ValueError:
                    numerical_value = None

        return numerical_value