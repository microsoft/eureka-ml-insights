from dataclasses import dataclass

import pandas as pd

from .transform import DFTransformBase


@dataclass
class GSM8KExtractAnswer(DFTransformBase):
    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.model_answer_column] = df[self.model_output_column].apply(self.parse_output_answer)
        return df

    @staticmethod
    def parse_output_answer(response: str) -> float:
        """
        Parse the input string to extract answer of a given GSM8K question.
        Parameters:
            response (str): Input string containing answer X
        Returns:
            numerical_value (float): A numeric value representing the model's answer.
        """

        def str_to_float(hyp):
            for remove_char in [",", "$", "%", "g"]:
                hyp = hyp.replace(remove_char, "")
            if hyp == "True" or hyp == "true":
                hyp = True
            if hyp == "False" or hyp == "false":
                hyp = False
            try:
                hyp = float(hyp)
            except ValueError:
                try:
                    hyp = eval(hyp)  # execute equations
                    if not ((type(hyp) is int) or (type(hyp) is float)):
                        hyp = None
                except Exception:
                    hyp = None
            return hyp

        if "####" not in response:
            return None

        answer = str(response).split("####")[-1].strip()
        numerical_value = str_to_float(answer)

        return numerical_value
