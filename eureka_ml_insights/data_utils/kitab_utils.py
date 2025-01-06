# Part of this file was authored by authors of the Kitab dataset (https://huggingface.co/datasets/microsoft/kitab)
# Some code in this file is copied from the original source repository and then adapted to fit this repository.
# The original license for this code is Community Data License Agreement - Permissive - Version 2.0

import ast
import re
import urllib.request
from dataclasses import dataclass

import pandas as pd

from .transform import DFTransformBase


@dataclass
class KitabExtractBooks(DFTransformBase):
    model_output_column: str
    model_books_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.model_books_column] = df[self.model_output_column].apply(parse_output_reason)
        return df


@dataclass
class GPT35KitabExtractBooks(KitabExtractBooks):
    model_output_column: str
    model_books_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        GPT 3.5 sometimes does not follow instructions and does not add the "Output:\n" marker in its format
        add_output_marker adds a "Output:\n" string to the model output prior to parsing, if it detects a list start, i.e. "1."
        """
        df[self.model_output_column] = df[self.model_output_column].apply(parse_output_reason(add_output_marker))
        return df


@dataclass
class PrepareContext(DFTransformBase):
    all_books_column: str
    all_books_context_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.all_books_context_column] = df[self.all_books_column].apply(prepare_all_books_context)
        return df


def prepare_all_books_context(s):
    """
    Parse the array of all books such that it is presented to the model in the following format:
    Title (year)    \n
    Title (year)
    """
    book_array = ast.literal_eval(s)
    context = ""
    for book in book_array:
        context += book + "    \n"
    return context


def add_output_marker(s):
    if s == None:
        s = "Output:\n"
    # Look for the last occurrence of 'Output:\n'
    last_output_index = s.rfind("Output:\n")
    if last_output_index == -1 and s.find("1.") == 0:
        s = "Output:\n" + s
    return s


# with all books
def parse_output_reason(s):
    """
    Parse the input string to extract titles and reasons from the 'Output:' section.
    Parameters:
    s (str): Input string containing information with 'Output:' section.
    Returns:
    dict: A dictionary containing extracted titles and reasons.
          Example: {'titles': ['Title 1', 'Title 2'], 'reasons': ['Reason 1', 'Reason 2']}
    """
    if s == None:
        return {"titles": [], "reasons": []}
    # Look for the last occurrence of 'Output:\n'
    last_output_index = s.rfind("Output:\n")

    # If 'All Books' is found in the string but 'Output:' is not found at all
    if "All Books" in s and last_output_index == -1:
        return {"titles": [], "reasons": []}

    # If found, only consider the text after this occurrence
    if last_output_index != -1:
        s = s[last_output_index + len("Output:\n") :]

    # regex for extracting reason and title
    reason_pattern = r"Reason: (.*?). Title:"

    # Adjust the title pattern to make year optional
    title_pattern = r"Title: (.*?)\s*(?:\(\d{4}\))?$"

    reasons = re.findall(reason_pattern, s, re.MULTILINE)
    titles = re.findall(title_pattern, s, re.MULTILINE)

    return {"titles": titles, "reasons": reasons}


def get_gpt4_preprocessed_names(original_path, download_path):
    """
    Download all book titles that have a human name as pre processed by gpt4
    """
    urllib.request.urlretrieve(original_path, download_path)
    gpt4_names = []
    human_name_gpt4_data = pd.read_csv(download_path)
    for entry in human_name_gpt4_data["human_name_books"].tolist():
        gpt4_names.extend(ast.literal_eval(entry)["titles"])
    return gpt4_names
