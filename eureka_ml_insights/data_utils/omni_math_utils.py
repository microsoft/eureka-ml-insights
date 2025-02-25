import math
import re
from dataclasses import dataclass

import pandas as pd

from .transform import DFTransformBase

# @staticmethod
def parse_output_answer(response):
    """
    Parse the input string to extract the model judgement.
    Parameters:
        response (str): Input string containing model judgement as '## Equivalence Judgement: X '.
    Returns: 
        dict: A dict of extracted final answer and model based judgement.
    """
    if response is None or response == '':
        return {}

    parts = response.split("## ")
    data = {}
    
    for part in parts[1:]:
        lines = part.strip().split("\n")
        title = lines[0].strip().replace('#', '').replace('*', '').lower()
        content = "\n".join(lines[1:]).strip()
        
        if title == "Justification":
            data[title] = content
        else:
            data[title] = lines[1].strip() if len(lines) > 1 else ''
    
    return data

@dataclass
class Omni_Math_ParseLabel(DFTransformBase):
    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.model_answer_column] = df[self.model_output_column].apply(self.extract_label)
        return df
    
    @staticmethod
    def extract_label(response):
        data = parse_output_answer(response)
        label = 'Equivalence Judgement'.lower()
        model_label = data[label] if label in data else ''
        numeric_label = math.nan
        if model_label.strip().replace('#', '').replace('*', '').lower() == 'true':
            numeric_label = 1
        elif model_label.strip().replace('#', '').replace('*', '').lower() == 'false':
            numeric_label = 0
        if numeric_label == math.nan:
            print(data[label], model_label, numeric_label)
        return numeric_label
    

@dataclass
class Omni_Math_ParseSolution(DFTransformBase):
    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.model_answer_column] = df[self.model_output_column].apply(self.extract_solution)
        return df
    
    @staticmethod
    def extract_solution(response):
        data = parse_output_answer(response)
        label = 'Student Final Answer'.lower()
        model_label = data[label] if label in data else ''
        return model_label

    