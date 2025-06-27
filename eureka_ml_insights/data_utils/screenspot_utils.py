import re
from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import pandas as pd

import ast

from .transform import DFTransformBase, MultiColumnTransform

def extract_bounding_box(model_output_raw, is_valid):
    """
    Extracts the answer from the text based on known model output patterns.
    Searches for both a letter and whole word answer and returns both as they are not
    always consistent.

    Args:
    - model_output_raw (str): The text containing the model's answer.
    - options (str): The list of options.
    - match_first (bool): If True, the first match is used for answer extraction when multiple matches are found.

    Returns:
    - str or None: The extracted answers, or empty strings if no answer could be extracted.
    """

    # replace common subsitutions in model outputs
    if not is_valid or not model_output_raw:
        return ""

    # Regular expression pattern to match the first bounding box in the format [x0,y0,x1,y1]
    pattern = r'\[\s*([-\d\.]+,\s*[-\d\.]+,\s*[-\d\.]+,\s*[-\d\.]+)\s*\]'

    # Search for the last match in the text
    matches = re.findall(pattern, model_output_raw)
    bbox = None
    if matches:
        # Get the last match and return as tuple of integers
        bbox = matches[-1]
        bbox = ast.literal_eval(bbox)
    return bbox

@dataclass
class ExtractBoundingBox(DFTransformBase):
    """This class is an answer extractor for the SCREENSPOT and SCREENSPOT_PRO benchmark."""

    answer_column_name: str
    extracted_answer_column_name: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.extracted_answer_column_name] = df.apply(
            lambda x: extract_bounding_box(x[self.answer_column_name], x["is_valid"]), axis=1
        )
        return df
