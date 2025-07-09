"""This module provides functions and classes for parsing output from a math model 
and applying transformations to data frames by extracting labels and solutions."""

import math
from dataclasses import dataclass

import pandas as pd

from .transform import DFTransformBase


def parse_output_answer(response):
    """Parses the input string to extract the model judgement.

    Args:
        response (str): Input string containing model judgement as '## Equivalence Judgement: X '.

    Returns:
        dict: A dictionary of extracted final answer and model-based judgement.
    """
    if response is None or response == "":
        return {}

    parts = response.split("## ")
    data = {}

    for part in parts[1:]:
        lines = part.strip().split("\n")
        title = lines[0].strip().replace("#", "").replace("*", "").lower()
        content = "\n".join(lines[1:]).strip()

        if title == "Justification".lower():
            data[title] = content
        else:
            data[title] = lines[1].strip() if len(lines) > 1 else ""

    return data


@dataclass
class Omni_Math_ParseLabel(DFTransformBase):
    """A DFTransformBase subclass that parses a math model's label output."""

    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the DataFrame by extracting labels from the model output.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame with the extracted labels appended.
        """
        df[self.model_answer_column] = df[self.model_output_column].apply(self.extract_label)
        return df

    @staticmethod
    def extract_label(response):
        """Extracts the numeric label from the model's response.

        Args:
            response (str): The model's output string.

        Returns:
            float: The numeric label representing the model's equivalence judgement
                (1 for True, 0 for False, NaN otherwise).
        """
        data = parse_output_answer(response)
        label = "equivalence judgement"
        model_label = data[label] if label in data else ""
        numeric_label = math.nan
        if model_label.strip().replace("#", "").replace("*", "").lower() == "true":
            numeric_label = 1
        elif model_label.strip().replace("#", "").replace("*", "").lower() == "false":
            numeric_label = 0
        if numeric_label == math.nan:
            print(data[label], model_label, numeric_label)
        return numeric_label


@dataclass
class Omni_Math_ParseSolution(DFTransformBase):
    """A DFTransformBase subclass that parses a math model's solution output."""

    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the DataFrame by extracting the solution from the model output.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame with the extracted solutions appended.
        """
        df[self.model_answer_column] = df[self.model_output_column].apply(self.extract_solution)
        return df

    @staticmethod
    def extract_solution(response):
        """Extracts the solution from the model's response.

        Args:
            response (str): The model's output string.

        Returns:
            str: The solution extracted from the model's response, or an empty string if not present.
        """
        data = parse_output_answer(response)
        label = "student final answer"
        model_label = data[label] if label in data else ""
        return model_label
