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

################# version 2 ######################

# def extract_final_answer(model_output):
#     """Extracts the final answer enclosed within <final_answer> tags."""
#     pattern = re.compile(r'<final_answer>(.*?)</final_answer>', re.DOTALL)
#     match = pattern.search(model_output)       
#     return match.group(1).strip() if match else None

# def extract_final_answer(model_output):
#     """Extracts the final answer enclosed within the last <final_answer> tags.
#        Returns None if no such tags are found."""
#     pattern = re.compile(r'<final_answer>(.*?)</final_answer>', re.DOTALL)
#     matches = pattern.findall(model_output)

#     # If no matches, return None
#     if not matches:
#         return None

#     # Return the last match, stripped of leading/trailing whitespace
#     return matches[-1].strip()

def extract_final_answer(model_output):
    """
    Searches from the end of the string for the last valid <final_answer>...</final_answer> pair.
    1. Finds the last occurrence of </final_answer>.
    2. From that position, searches backward for the most recent <final_answer>.
    3. If found, extracts and returns the text inside. Otherwise, keeps going backward.
    4. Returns None if no valid pairs are found.
    """
    start_tag = '<final_answer>'
    end_tag = '</final_answer>'

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
    solution = solution_dict.get('Solution')

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


def parse_path_from_model_output(model_output_string):
    """Parses the model output to extract a SAT path."""    
    # print(model_output_string)
    final_answer = extract_final_answer(model_output_string)

    # print("final_answer: ", final_answer)

    sat_solution=""

    try:
        sat_solution = extract_solution(final_answer) if final_answer else None
    except Exception as e:
        print("\n\n\n\n\n")
        print("final_answer: ", final_answer)
        print(f"An error occurred while extracting the solution: {e}")
        sat_solution = None

    
    if sat_solution is None:
        return "-1"

    binary_soln_string = ""

    if sat_solution:
        binary_soln_string = convert_to_binary_string(sat_solution)

    # print("binary_soln_string: ", binary_soln_string)

    return binary_soln_string





########################################
### version 1 ##########################
########################################

# import json
# import re
# from dataclasses import dataclass

# import pandas as pd

# from eureka_ml_insights.data_utils import DFTransformBase
# import ast


# @dataclass
# class NPHARDSATExtractAnswer(DFTransformBase):
#     """Class to extract and transform the SAT path from model output."""

#     model_output_column: str
#     model_answer_column: str

#     def transform(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Extracts the SAT path from the model output and stores it in the model_answer_column."""
#         df[self.model_answer_column] = df[self.model_output_column].apply(parse_path_from_model_output)
#         return df

# def extract_final_answer(model_output):
#     """Extracts the final answer enclosed within <final_answer> tags."""
#     pattern = re.compile(r'<final_answer>(.*?)</final_answer>', re.DOTALL)
#     match = pattern.search(model_output)       
#     return match.group(1).strip() if match else None

# def extract_solution(final_answer):
#     solution_dict = ast.literal_eval(final_answer)    
#     solution = solution_dict.get('Solution')

#     return solution

# def convert_to_binary_string(solution_str):
    
#     if solution_str == "Unsatisfiable":
#         return ""  # Return empty string if the solution is "Unsatisfactory"

#     # 1. Split the string on commas.
#     parts = solution_str.split(",")

#     # 2. Convert each element to '1' if it's 'True', else '0'.
#     converted_parts = []
#     for p in parts:
#         p = p.strip()  # remove extra whitespace
#         if p == "True":
#             converted_parts.append("1")
#         else:
#             converted_parts.append("0")

#     # 3. Join them back together as a comma-separated string.
#     converted_str = ",".join(converted_parts)

#     return converted_str


# def parse_path_from_model_output(model_output_string):
#     """Parses the model output to extract a SAT path."""    

#     final_answer = extract_final_answer(model_output_string)

#     try:
#         sat_solution = extract_solution(final_answer) if final_answer else None
#     except Exception as e:
#         print(f"An error occurred while extracting the solution: {e}")
#         sat_solution = None
    
#     binary_soln_string = ""

#     if sat_solution:
#         binary_soln_string = convert_to_binary_string(sat_solution)

#     return binary_soln_string
