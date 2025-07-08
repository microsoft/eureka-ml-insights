"""
This module provides utilities to parse model outputs for SAT solutions and convert
them to binary representations. It includes a data transform class that extracts the
SAT assignment from the model output and several helper functions for parsing and
converting the solution.
"""

import ast
import logging
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from eureka_ml_insights.data_utils import DFTransformBase


@dataclass
class NPHARDSATExtractAnswer(DFTransformBase):
    """
    Extracts and transforms the SAT path from model output.

    Attributes:
        model_output_column (str): The name of the column containing the model's raw output.
        model_answer_column (str): The name of the column where the transformed answer is stored.
    """

    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts the SAT assignment from the model output and stores it in the specified column.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The dataframe with the extracted SAT assignment stored in the
            'model_answer_column' column.
        """
        df[self.model_answer_column] = df[self.model_output_column].apply(parse_assignment_from_model_output)
        return df


def extract_final_answer(model_output):
    """
    Retrieves the text within the last <final_answer>...</final_answer> block from the model output.

    Args:
        model_output (str): The raw output from the model, which may contain <final_answer> blocks.

    Returns:
        Optional[str]: The content within the final <final_answer>...</final_answer> block,
        or None if no block is found.
    """
    open_tag = "<final_answer>"
    last_open = model_output.rfind(open_tag)
    if last_open == -1:
        return None
    sliced = model_output[last_open:]
    matches = re.findall(r"<final_answer>(.*?)</final_answer>", sliced, re.DOTALL)
    return matches[-1] if matches else None


def extract_solution(final_answer: str) -> Optional[str]:
    """
    Parses a final answer string (for example, "{'Solution': 'True, False, ...'}") and
    returns the value of the 'Solution' key.

    Args:
        final_answer (str): A string representation of a dictionary containing a 'Solution' key.

    Returns:
        Optional[str]: The assignment string if present and well-formed, otherwise None.
    """
    try:
        parsed = ast.literal_eval(final_answer)
    except ValueError as err:
        logging.info(f"extract_solution: literal_eval failed: {err}")
        return None

    try:
        return parsed.get("Solution")
    except AttributeError:
        logging.info(f"extract_solution: expected a dict-like object but got {type(parsed).__name__}")
    except KeyError:
        logging.info("extract_solution: 'Solution' key not found in parsed result")

    return None


def convert_to_binary_string(solution):
    """
    Converts a comma-separated list of "True"/"False" flags to a comma-separated
    list of "1"/"0".

    Any token other than exactly "True" or "False" is converted to "-1".

    Example:
        >>> convert_to_binary_string("True, False, True, True")
        '1,0,1,1'

    Args:
        solution (str): A comma-separated string containing "True" or "False" values.

    Returns:
        str: A comma-separated string of "1" and "0", or "-1" if invalid input is encountered.
    """
    if not isinstance(solution, str):
        return "-1"

    parts = [p.strip() for p in solution.split(",")]
    if not all(p in ["True", "False"] for p in parts):
        return "-1"

    converted_parts = ["1" if p == "True" else "0" for p in parts]
    return ",".join(converted_parts)


def parse_assignment_from_model_output(model_output: str) -> str:
    """
    Extracts a SAT assignment from a model's raw output and converts it to a binary string.

    Args:
        model_output (str): The raw output from the model.

    Returns:
        str: A binary string representing the assignment, or "-1" if no valid solution is found.
    """
    final_answer: Optional[str] = extract_final_answer(model_output)
    if not final_answer:
        logging.info("No final answer section detected in model output.")
        return "-1"

    sat_solution = extract_solution(final_answer)
    if not sat_solution:
        return "-1"

    if sat_solution == "Unsatisfiable":
        return ""

    return convert_to_binary_string(sat_solution)
