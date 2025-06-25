"""This module provides functionality for extracting TSP paths from model output.

It includes:
1) A data transform class (NPHARDTSPExtractAnswer) that organizes extraction in a DataFrame.
2) Utility functions (extract_final_answer, extract_path, parse_path_from_model_output) to parse TSP paths from raw strings.
"""

import json
import logging
import re
from dataclasses import dataclass

import pandas as pd

from eureka_ml_insights.data_utils import DFTransformBase


@dataclass
class NPHARDTSPExtractAnswer(DFTransformBase):
    """Extracts TSP paths from model output columns in a DataFrame.

    This class transforms the TSP data in a DataFrame by applying the
    parse_path_from_model_output function to each row.

    Attributes:
        model_output_column (str): The name of the column containing model outputs.
        model_answer_column (str): The name of the column to store extracted TSP paths.
    """

    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the DataFrame by extracting the TSP path.

        This method applies parse_path_from_model_output to the specified
        model output column and stores the result in the specified
        model answer column.

        Args:
            df (pd.DataFrame): The DataFrame containing the model output.

        Returns:
            pd.DataFrame: The transformed DataFrame with the extracted TSP path.
        """
        df[self.model_answer_column] = df[self.model_output_column].apply(parse_path_from_model_output)
        return df


def extract_final_answer(model_output):
    """Extracts the final answer from the model output.

    Finds all non-overlapping occurrences between <final_answer> and </final_answer>
    and returns the last occurrence if any are found, otherwise returns None.

    Args:
        model_output (str): The raw output string from the model.

    Returns:
        str or None: The last occurrence of the final answer if found, otherwise None.
    """
    matches = re.findall(r"<final_answer>(.*?)</final_answer>", model_output, flags=re.DOTALL)
    return matches[-1] if matches else None


def extract_path(final_answer):
    """Extracts the path string from the final answer.

    This function tries to parse the final answer as JSON or uses a regex fallback
    to extract the "Path" value.

    Args:
        final_answer (str): The final answer string extracted from model output.

    Returns:
        str or None: The path string if extracted, otherwise None.
    """
    try:
        # Convert single quotes to double quotes for valid JSON parsing
        final_answer_json = json.loads(final_answer.replace("'", '"'))
        return final_answer_json.get("Path", None)
    except json.JSONDecodeError:
        # Fallback regex extraction if JSON parsing fails
        match = re.search(r'"Path":\s*"([^"]+)"', final_answer)
        return match.group(1) if match else None


def parse_path_from_model_output(model_output_string):
    """Parses the model output to extract a TSP path.

    This function extracts the final answer, then attempts to retrieve
    the path and parse it into a list of integers. If no valid path is
    found, a default path "0,0,0,0" is returned.

    Args:
        model_output_string (str): The raw output string from the model.

    Returns:
        str: A comma-separated string of integers indicating the TSP path.
    """
    try:
        final_answer = extract_final_answer(model_output_string)
        tour_string = extract_path(final_answer) if final_answer else None

        if tour_string is None:
            return "0,0,0,0"

        parts = re.findall(r"\d+|->", tour_string)
        tour_string = "".join(parts)
        tour = list(map(int, tour_string.split("->")))
    except (AttributeError, ValueError) as e:
        logging.info(f"There is no valid path: {e}")
        return "0,0,0,0"

    return ",".join(map(str, tour))