"""Module for extracting GSM8K answers from model output and storing them in a DataFrame."""

from dataclasses import dataclass

import pandas as pd

from .transform import DFTransformBase


@dataclass
class GSM8KExtractAnswer(DFTransformBase):
    """Extracts numeric answers from GSM8K model output columns in a DataFrame.

    Attributes:
        model_output_column (str): The name of the column containing the raw model output.
        model_answer_column (str): The name of the column to store the extracted numeric answer.
    """

    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the input DataFrame by extracting numeric answers from model output.

        Args:
            df (pd.DataFrame): The input DataFrame containing model output data.

        Returns:
            pd.DataFrame: A DataFrame with an additional column storing the extracted numeric answers.
        """
        df[self.model_answer_column] = df[self.model_output_column].apply(self.parse_output_answer)
        return df

    @staticmethod
    def parse_output_answer(response: str) -> float:
        """Extracts a numeric answer from a given GSM8K response string.

        This function looks for the answer text after "####", strips unwanted
        characters, and attempts to convert it to a float. If the conversion fails,
        it returns None.

        Args:
            response (str): The raw string response containing the answer.

        Returns:
            float: The numeric value of the answer, or None if parsing fails.
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
