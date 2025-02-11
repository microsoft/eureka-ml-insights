import json
import logging
import re
from dataclasses import dataclass

import pandas as pd

from eureka_ml_insights.data_utils import DFTransformBase


@dataclass
class NPHARDTSPExtractAnswer(DFTransformBase):
    """Class to extract and transform the TSP path from model output."""

    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts the tsp path from the model output and stores it in the model_answer_column."""
        df[self.model_answer_column] = df[self.model_output_column].apply(parse_path_from_model_output)
        return df


def extract_final_answer(model_output):
    # Find all non-overlapping occurrences between <final_answer> and </final_answer>
    matches = re.findall(r"<final_answer>(.*?)</final_answer>", model_output, flags=re.DOTALL)

    # Return the last occurrence if any are found, otherwise return None
    return matches[-1] if matches else None


def extract_path(final_answer):
    """Extracts the path string from the final answer, handling both JSON formats."""
    try:
        # Convert single quotes to double quotes for valid JSON parsing
        final_answer_json = json.loads(final_answer.replace("'", '"'))
        return final_answer_json.get("Path", None)
    except json.JSONDecodeError:
        # Fallback regex extraction if JSON parsing fails
        match = re.search(r'"Path":\s*"([^"]+)"', final_answer)
        return match.group(1) if match else None


def parse_path_from_model_output(model_output_string):
    """Parses the model output to extract a tsp path."""
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
