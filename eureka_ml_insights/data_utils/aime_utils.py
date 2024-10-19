# Part of this file was authored by authors of the Kitab dataset (https://huggingface.co/datasets/microsoft/kitab)
# Some code in this file is copied from the original source repository and then adapted to fit this repository.
# The original license for this code is Community Data License Agreement - Permissive - Version 2.0

import ast
import re
import urllib.request
from dataclasses import dataclass

import pandas as pd

from eureka_ml_insights.data_utils import DFTransformBase


@dataclass
class AIMEExtractAnswer(DFTransformBase):
    model_output_column: str
    model_books_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.model_books_column] = df[self.model_output_column].apply(parse_output_answer)
        return df

# with all books
def parse_output_answer(response):
    """
    Parse the input string to extract titles and reasons from the 'Output:' section.
    Parameters:
    s (str): Input string containing information with 'Output:' section.
    Returns:
    dict: A dictionary containing extracted titles and reasons.
          Example: {'titles': ['Title 1', 'Title 2'], 'reasons': ['Reason 1', 'Reason 2']}
    """
    # Try to find an answer in the "Final Answer: X" format
    match = re.search(r"Final Answer:\s*([\$]?[\d,]+(?:\.\d+)?%?)", response)
    if match:
        answer_str = match.group(1)
    else:
        # If that fails, look for any number with optional $ or % in the response
        match = re.search(r"([\$]?[\d,]+(?:\.\d+)?%?)", response)
        answer_str = match.group(1) if match else None

    numerical_value = None
    # Store the original format
    original_format = answer_str

    if answer_str:
        # Remove $ and commas, handle percentages for numerical comparison
        answer_str = answer_str.replace("$", "").replace(",", "")
        if answer_str.endswith("%"):
            numerical_value = (
                float(answer_str[:-1]) / 100
            )  # Convert percentage to decimal
        else:
            try:
                numerical_value = float(answer_str)
            except ValueError as e:
                print(f"Error converting answer '{answer_str}' to numerical value: {e}")
                numerical_value = None

    if numerical_value is not None:
        print(f"Extracted Answer: {original_format}")
        print(f"Numerical Value for Comparison: {numerical_value}")
    else:
        print("No valid answer extracted. use 0")
        #numerical_value = 0

    return numerical_value