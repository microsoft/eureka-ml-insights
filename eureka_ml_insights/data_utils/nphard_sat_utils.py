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

################################

def extract_final_answer3(model_output):
    """Extracts the final answer enclosed within <final_answer> tags."""
    pattern = re.compile(r'<final_answer>(.*?)</final_answer>', re.DOTALL)
    match = pattern.search(model_output)       
    return match.group(1).strip() if match else None

def extract_solution3(final_answer):
    print("solution_dict1: ", final_answer)
    solution_dict = ast.literal_eval(final_answer)
    print("solution_dict2: ", solution_dict)
    solution = solution_dict.get('Solution')

    return solution

def convert_to_binary_string(solution_str):
    print("convert to binary soln_str: ", solution_str)
    if solution_str == "Unsatisfiable":
        return ""  # Return empty string if the solution is "Unsatisfactory"
    # breakpoint()
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

##############################



def parse_path_from_model_output(model_output_string):
    """Parses the model output to extract a SAT path."""    
    # print("model_output_string: ", model_output_string)
    final_answer = extract_final_answer3(model_output_string)

    print("final_answer: ", final_answer)

    sat_solution = extract_solution3(final_answer) if final_answer else None
    
    print("sat_solution: ", sat_solution)

    binary_soln_string = ""

    if sat_solution:
        binary_soln_string = convert_to_binary_string(sat_solution)

    print("binary_sat_solution: ", binary_soln_string)

    return binary_soln_string


#############################


# def extract_final_answer(model_output):
#     """Extracts the final answer enclosed within <final_answer> tags."""
#     match = re.search(r"<final_answer>(.*?)</final_answer>", model_output, re.DOTALL)
#     return match.group(1).strip() if match else None


# def extract_path(final_answer):
#     """Extracts the path string from the final answer, handling both JSON formats."""
#     try:
#         # Convert single quotes to double quotes for valid JSON parsing
#         final_answer_json = json.loads(final_answer.replace("'", '"'))
#         return final_answer_json.get("Solution", None)
#     except json.JSONDecodeError:
#         # Fallback regex extraction if JSON parsing fails
#         # match = re.search(r"<final_answer>\{'Solution':\s*'([^']+)'\}</final_answer>", text)
#         match = re.search(r'"Solution":\s*"([^"]+)"', final_answer)
#         return match.group(1) if match else None

# def extract_solution(final_answer):
#     if isinstance(final_answer, str):
#         solution = final_answer
#     elif isinstance(final_answer, dict):
#         solution = final_answer.get('Solution', final_answer)  # Handle direct dict case
#     else:
#         return {}
    
#     if isinstance(solution, dict):
#         return {k.replace('_', ''): v for k, v in solution.items()}  # Normalize variable names
    
#     extracted_values = {}
#     matches = re.findall(r"(x\d+|x_\d+)\s*=\s*(True|False)", solution)
    
#     for var, value in matches:
#         normalized_var = var.replace('_', '')  # Normalize variable names
#         extracted_values[normalized_var] = value == 'True'
    
#     return extracted_values


# def extract_solution2(final_answer):
#     # Convert string to dictionary
#     parsed_dict = ast.literal_eval(final_answer)

#     # Extract Solution values
#     solution_values = parsed_dict.get('Solution', {})

#     # Handling different formats
#     extracted_values = {}

#     if isinstance(solution_values, dict):
#         # Case 1: Solution is already a dictionary
#         extracted_values = {k.replace('_', ''): v for k, v in solution_values.items()}
#     elif isinstance(solution_values, str):
#         # Case 2: Solution is a string, parse it
#         for pair in solution_values.split(','):
#             key, value = pair.strip().split('=')
#             formatted_key = key.strip().replace('_', '')  # Remove underscore
#             extracted_values[formatted_key] = value.strip() == 'True'

#     # Print extracted values
#     formatted_output = ', '.join([f"'{k}'= {v}" for k, v in extracted_values.items()])

#     return formatted_output

# def convert_to_binary_string(solution):
#     if solution == "Unsatisfactory":
#         return ""  # Return empty string if the solution is "Unsatisfactory"

#     # Split by comma and handle cases with or without spaces
#     variables = re.split(r',\s*', solution)  

#     binary_representation = []
#     for var in variables:
#         key_value = re.split(r'\s*=\s*', var)  # Handle cases with or without spaces around '='
#         if len(key_value) == 2:
#             binary_representation.append(str(int(key_value[1] == 'True')))

#     return ','.join(binary_representation)
