import re, json, ast
from dataclasses import dataclass

import pandas as pd

from .transform import DFTransformBase


@dataclass
class BFCLExtractAnswer(DFTransformBase):
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
        try:
            json.loads(response)
            return response  # valid JSON
        except json.JSONDecodeError:
            try:
                result = ast.literal_eval(response)
                if isinstance(result, dict):
                    return json.dumps(result)
            except Exception:
                pass
        return "{}"
