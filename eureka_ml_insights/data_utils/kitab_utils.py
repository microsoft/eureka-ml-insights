# Part of this file was authored by authors of the Kitab dataset (https://huggingface.co/datasets/microsoft/kitab)
# Some code in this file is copied from the original source repository and then adapted to fit this repository.
# The original license for this code is Community Data License Agreement - Permissive - Version 2.0

"""
Module for parsing and transforming book data from the Kitab dataset.

This module provides classes and functions to parse model outputs, extract book
information, and prepare context data. It includes utilities to handle specific
modifications and standard output marker additions for more consistent data
parsing. Additionally, it provides a method to retrieve GPT-4 preprocessed names
from a specified dataset.
"""

import ast
import re
import urllib.request
from dataclasses import dataclass

import pandas as pd

from .transform import DFTransformBase


@dataclass
class KitabExtractBooks(DFTransformBase):
    """
    Class for extracting books from a DataFrame by parsing model outputs.

    Attributes:
        model_output_column (str): The column name containing model output.
        model_books_column (str): The column name to store extracted book data.
    """
    model_output_column: str
    model_books_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame by parsing the specified model output column
        to extract book information.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame with the extracted book data.
        """
        df[self.model_books_column] = df[self.model_output_column].apply(parse_output_reason)
        return df


@dataclass
class KitabExtractBooksAddMarker(DFTransformBase):
    """
    Class for extracting books from a DataFrame by parsing model outputs, with an
    added step for injecting an 'Output:' marker if needed.

    Attributes:
        model_output_column (str): The column name containing model output.
        model_books_column (str): The column name to store extracted book data.
    """
    model_output_column: str
    model_books_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame by adding an output marker if needed and then
        parsing the specified model output column to extract book information.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame with the extracted book data.
        """
        df[self.model_books_column] = df[self.model_output_column].apply(add_output_marker)
        df[self.model_books_column] = df[self.model_books_column].apply(parse_output_reason)
        return df


@dataclass
class PrepareContext(DFTransformBase):
    """
    Class for preparing context data for a set of books.

    Attributes:
        all_books_column (str): The column name containing all books data.
        all_books_context_column (str): The column name to store the prepared
            context data.
    """
    all_books_column: str
    all_books_context_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame by preparing the context for all books in the
        specified column.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame with prepared context data.
        """
        df[self.all_books_context_column] = df[self.all_books_column].apply(prepare_all_books_context)
        return df


def prepare_all_books_context(s):
    """
    Parses the array of all books and presents it in a specific text format.

    This function reads the string representation of a list of books, and
    returns a string in which each book is followed by four spaces and a
    newline.

    Args:
        s (str): The string representing a list of books.

    Returns:
        str: A formatted string of book titles, each on its own line.
    """
    book_array = ast.literal_eval(s)
    context = ""
    for book in book_array:
        context += book + "    \n"
    return context


def add_output_marker(s):
    """
    Adds an 'Output:\\n' marker to the given string if needed.

    If the string does not contain 'Output:\\n', and it starts with '1.', this
    function will prepend 'Output:\\n' to the string.

    Args:
        s (str): The input string.

    Returns:
        str: The possibly modified string with an output marker.
    """
    if s == None:
        s = "Output:\n"
    # Look for the last occurrence of 'Output:\n' with an arbitrary number of spaces
    pattern = r"Output:\s*\n"
    last_output_index = None
    for match in re.finditer(pattern, s):
        last_output_index = match.end()

    if last_output_index == -1 and s.find("1.") == 0:
        s = "Output:\n" + s
    return s


def parse_output_reason(s):
    """
    Parses the input string to extract titles and reasons from the 'Output:' section.

    The function looks for the substring 'Output:\\n', and if found, processes
    the subsequent text to extract pairs of reason and title. The extracted data
    is returned as a dictionary with keys 'titles' and 'reasons'.

    Args:
        s (str): The input string containing information with an 'Output:' section.

    Returns:
        dict: A dictionary with keys 'titles' and 'reasons', each containing a list
            of extracted values. Example:
            {'titles': ['Title 1', 'Title 2'], 'reasons': ['Reason 1', 'Reason 2']}
    """
    if s == None:
        return {"titles": [], "reasons": []}
    pattern = r"Output:\s*\n"
    last_output_index = None
    for match in re.finditer(pattern, s):
        last_output_index = match.end()

    # If 'All Books' is found in the string but 'Output:' is not found at all
    if "All Books" in s and last_output_index == -1:
        return {"titles": [], "reasons": []}

    # If found, only consider the text after this occurrence
    if last_output_index != -1:
        s = s[last_output_index:]

    # regex for extracting reason and title
    reason_pattern = r"Reason: (.*?). Title:"
    title_pattern = r"Title: (.*?)\s*(?:\(\d{4}\))?$"

    reasons = re.findall(reason_pattern, s, re.MULTILINE)
    titles = re.findall(title_pattern, s, re.MULTILINE)

    return {"titles": titles, "reasons": reasons}


def get_gpt4_preprocessed_names(original_path, download_path):
    """
    Downloads book titles preprocessed by GPT-4 that contain human names.

    This function retrieves a CSV file from the provided URL, reads the data, and
    collects book titles from a JSON-like column containing a dictionary with
    a list of 'titles'.

    Args:
        original_path (str): The URL or path to download the CSV from.
        download_path (str): The local path to store the downloaded CSV.

    Returns:
        list: A list of book titles extracted from the downloaded CSV.
    """
    urllib.request.urlretrieve(original_path, download_path)
    gpt4_names = []
    human_name_gpt4_data = pd.read_csv(download_path)
    for entry in human_name_gpt4_data["human_name_books"].tolist():
        gpt4_names.extend(ast.literal_eval(entry)["titles"])
    return gpt4_names