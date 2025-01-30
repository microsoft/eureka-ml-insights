import json
import re
from dataclasses import dataclass

import pandas as pd

from eureka_ml_insights.data_utils import DFTransformBase
import ast


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
    pattern = re.compile(r'<final_answer>(.*?)</final_answer>', re.DOTALL)
    match = pattern.search(model_output)       
    return match.group(1).strip() if match else None

def extract_solution(final_answer):
    solution_dict = ast.literal_eval(final_answer)    
    solution = solution_dict.get('Solution')

    return solution

def convert_to_binary_string(solution_str):
    
    if solution_str == "Unsatisfiable":
        return ""  # Return empty string if the solution is "Unsatisfactory"

    # 1. Split the string on commas.
    parts = solution_str.split(",")

    # 2. Convert each element to '1' if it's 'True', else '0'.
    converted_parts = []
    for p in parts:
        p = p.strip()  # remove extra whitespace
        if p == "True":
            converted_parts.append("1")
        else:
            converted_parts.append("0")

    # 3. Join them back together as a comma-separated string.
    converted_str = ",".join(converted_parts)

    return converted_str


def parse_path_from_model_output(model_output_string):
    """Parses the model output to extract a SAT path."""    

    final_answer = extract_final_answer(model_output_string)

    try:
        sat_solution = extract_solution(final_answer) if final_answer else None
    except Exception as e:
        print(f"An error occurred while extracting the solution: {e}")
        sat_solution = None
    
    binary_soln_string = ""

    if sat_solution:
        binary_soln_string = convert_to_binary_string(sat_solution)

    return binary_soln_string
