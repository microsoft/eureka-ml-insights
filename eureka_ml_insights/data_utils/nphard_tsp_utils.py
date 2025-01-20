import re
from dataclasses import dataclass
import ast
import pandas as pd
from eureka_ml_insights.data_utils import DFTransformBase
import xml.etree.ElementTree as ET

@dataclass
class NPHARDTSPExtractAnswer(DFTransformBase):
    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.model_answer_column] = df[self.model_output_column].apply(parse_path_from_model_output)
        return df

def extract_final_answer(model_output):
    match = re.search(r'<final_answer>(.*?)</final_answer>', model_output, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_path(final_answer):
    match = re.search(r"'Path':\s*'([^']+)'", final_answer)
    return match.group(1) if match else None

def parse_path_from_model_output(model_output_string):    
    final_answer = extract_final_answer(model_output_string)
    tour_string = extract_path(final_answer) if final_answer else None

    if tour_string is None:
        return "0,0,0,0"

    tour_string = re.sub(r'[^0-9->]', '', tour_string)
    tour = list(map(int, tour_string.split('->')))

    return ','.join(map(str, tour))