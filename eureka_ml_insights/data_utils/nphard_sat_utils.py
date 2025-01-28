import json
import re
from dataclasses import dataclass

import pandas as pd

from eureka_ml_insights.data_utils import DFTransformBase


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
    """Extracts the final answer enclosed within <final_answer> tags."""
    match = re.search(r"<final_answer>(.*?)</final_answer>", model_output, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_path(final_answer):
    """Extracts the path string from the final answer, handling both JSON formats."""
    try:
        # Convert single quotes to double quotes for valid JSON parsing
        final_answer_json = json.loads(final_answer.replace("'", '"'))
        return final_answer_json.get("Solution", None)
    except json.JSONDecodeError:
        # Fallback regex extraction if JSON parsing fails
        # match = re.search(r"<final_answer>\{'Solution':\s*'([^']+)'\}</final_answer>", text)
        match = re.search(r'"Solution":\s*"([^"]+)"', final_answer)
        return match.group(1) if match else None

def convert_to_binary_string(solution):
    if solution == "Unsatisfactory":
        return ""  # Return empty string if the solution is "Unsatisfactory"

    # Split by comma and handle cases with or without spaces
    variables = re.split(r',\s*', solution)  

    binary_representation = []
    for var in variables:
        key_value = re.split(r'\s*=\s*', var)  # Handle cases with or without spaces around '='
        if len(key_value) == 2:
            binary_representation.append(str(int(key_value[1] == 'True')))

    return ','.join(binary_representation)

def parse_path_from_model_output(model_output_string):
    """Parses the model output to extract a SAT path."""    
    final_answer = extract_final_answer(model_output_string)
    sat_solution = extract_path(final_answer) if final_answer else None
    
    binary_soln_string = ""

    if sat_solution:
        binary_soln_string = convert_to_binary_string(sat_solution)

    return binary_soln_string
