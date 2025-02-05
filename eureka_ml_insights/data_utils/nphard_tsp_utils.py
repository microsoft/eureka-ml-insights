import json
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

        # Remove non-numeric characters except '->' and split into a list of integers
        parts = re.findall(r"\d+|->", tour_string)        
        tour_string = "".join(parts)
        tour = list(map(int, tour_string.split("->")))            
    except Exception as e:                
        return "0,0,0,0"

    return ",".join(map(str, tour))