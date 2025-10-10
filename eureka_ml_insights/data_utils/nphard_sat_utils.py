import ast
import logging
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from eureka_ml_insights.data_utils import DFTransformBase


@dataclass
class NPHARDSATExtractAnswer(DFTransformBase):
    """Class to extract and transform the SAT path from model output."""

    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts the SAT assignment from the model output and stores it in the model_answer_column."""
        df[self.model_answer_column] = df[self.model_output_column].apply(parse_assignment_from_model_output)
        return df


def extract_final_answer(model_output):
    """Return the text inside the *last* <final_answer> … </final_answer> block."""
    open_tag = "<final_answer>"
    # Locate the last occurrence of the opening tag.
    last_open = model_output.rfind(open_tag)
    if last_open == -1:
        return None
    sliced = model_output[last_open:]
    # Grab everything between <final_answer> and </final_answer>
    matches = re.findall(r"<final_answer>(.*?)</final_answer>", sliced, re.DOTALL)

    # Return the final captured block (if any); otherwise None.
    return matches[-1] if matches else None


def extract_solution(final_answer: str) -> Optional[str]:
    """
    Parse ``final_answer`` (which should look like ``{'Solution': 'True, False, ...'}``)
    and return the value of the ``"Solution"`` key.

    Returns
    -------
    Optional[str]
        The assignment string if present and well-formed, otherwise ``None``.
    """
    # Try to turn the raw string into a Python object.
    try:
        parsed = ast.literal_eval(final_answer)
    except (ValueError, SyntaxError) as err:
        logging.info(f"extract_solution: literal_eval failed: {err}")
        return None

    # Ensure we really got something dict-like.
    try:
        return parsed.get("Solution")
    except AttributeError:
        logging.info("extract_solution: expected a dict-like object but got " f"{type(parsed).__name__}")
    except KeyError:
        logging.info("extract_solution: 'Solution' key not found in parsed result")

    return None


def convert_to_binary_string(solution):
    """
    Convert a comma-separated list of “True”/“False” flags into a
    comma-separated list of “1”/“0”.

    Special cases
    -------------
    * Any token other than exactly "True" or "False" →  "-1"

    Example
    -------
    >>> convert_to_binary_string("True, False, True, True")
    '1,0,1,1'
    """
    # If the solution is not a string, there's nothing to process
    if not isinstance(solution, str):
        return "-1"

    # Split on commas and strip whitespace
    parts = [p.strip() for p in solution.split(",")]

    # If parts are not strictly "True" or "False", return -1.
    if not all(p in ["True", "False"] for p in parts):
        return "-1"

    # Convert "True" -> "1" and "False" -> "0"
    converted_parts = ["1" if p == "True" else "0" for p in parts]

    # Join the converted parts with commas
    return ",".join(converted_parts)


def parse_assignment_from_model_output(model_output: str) -> str:
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
    if model_output is None or not isinstance(model_output, str) or not model_output.strip():
        logging.info("Model output is None or empty or not a string.")
        return "-1"
    final_answer: Optional[str] = extract_final_answer(model_output)

    if not final_answer:
        logging.info(f"No final answer section detected in model output.")
        return "-1"

    # Try to parse the SAT solution from the answer block.
    sat_solution = extract_solution(final_answer)

    if not sat_solution:
        return "-1"

    # If the solution is "Unsatisfiable"
    if sat_solution == "Unsatisfiable":
        return ""

    # Convert parsed solution to a binary string representation.
    return convert_to_binary_string(sat_solution)
