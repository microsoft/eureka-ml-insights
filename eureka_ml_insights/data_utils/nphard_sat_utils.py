import ast
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from eureka_ml_insights.data_utils import DFTransformBase

logger = logging.getLogger(__name__)


@dataclass
class NPHARDSATExtractAnswer(DFTransformBase):
    """Class to extract and transform the SAT path from model output."""

    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts the SAT path from the model output and stores it in the model_answer_column."""
        df[self.model_answer_column] = df[self.model_output_column].apply(parse_path_from_model_output)
        return df


def extract_final_answer(model_output):
    """
    Searches from the end of the string for the last valid <final_answer>...</final_answer> pair.
    1. Finds the last occurrence of </final_answer>.
    2. From that position, searches backward for the most recent <final_answer>.
    3. If found, extracts and returns the text inside. Otherwise, keeps going backward.
    4. Returns None if no valid pairs are found.
    """
    start_tag = "<final_answer>"
    end_tag = "</final_answer>"

    search_upto = len(model_output)
    while True:
        # 1. Find the last occurrence of </final_answer> up to 'search_upto'
        end_index = model_output.rfind(end_tag, 0, search_upto)
        if end_index == -1:
            # No further </final_answer> found: no valid pairs exist
            return None

        # 2. Find the last <final_answer> that appears before this </final_answer>
        start_index = model_output.rfind(start_tag, 0, end_index)
        if start_index == -1:
            # There's a </final_answer> but no preceding <final_answer> to match
            # Move 'search_upto' to the position of this </final_answer> so we ignore it
            # and keep searching earlier in the string.
            search_upto = end_index
            continue

        # 3. We have a valid pair: extract the substring inside the tags
        start_content = start_index + len(start_tag)
        content = model_output[start_content:end_index]

        return content.strip()


def extract_solution(final_answer):
    solution_dict = ast.literal_eval(final_answer)
    solution = solution_dict.get("Solution")

    return solution


def convert_to_binary_string(solution):
    # If the solution is the special Ellipsis object (...)
    if solution is Ellipsis:
        return "-1"

    # If the solution is not a string, there's nothing to process
    if not isinstance(solution, str):
        return "-1"

    # If the solution is "Unsatisfiable"
    if solution == "Unsatisfiable":
        return ""

    # Split on commas and strip whitespace
    parts = [p.strip() for p in solution.split(",")]

    # If parts are not strictly "True" or "False", return empty string
    if not all(p in ["True", "False"] for p in parts):
        return "-1"

    # Convert "True" -> "1" and "False" -> "0"
    converted_parts = ["1" if p == "True" else "0" for p in parts]

    # Join the converted parts with commas
    return ",".join(converted_parts)


def parse_path_from_model_output(model_output: str) -> str:
    """
    Extract a SAT assignment from a model's raw output and convert it to a
    binary string.

    Returns
    -------
    str
        A binary string representing the assignment, or "-1" if no valid
        solution is found.
    """
    # Pull out the “final answer” block the model produced.
    final_answer: Optional[str] = extract_final_answer(model_output)

    if not final_answer:
        logger.info("No final answer section detected in model output.")
        return "-1"

    # Try to parse the SAT solution from the answer block.
    try:
        sat_solution = extract_solution(final_answer)
    except (AttributeError, ValueError) as exc:
        logger.info("Failed to parse SAT solution: %s", exc)
        return "-1"

    if not sat_solution:
        return "-1"

    # Convert parsed solution to a binary string representation.
    return convert_to_binary_string(sat_solution)
