import re
from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import pandas as pd

from .transform import DFTransformBase, MultiColumnTransform


@dataclass
class LowerCaseNoPunctuationConvertNumbers(MultiColumnTransform):
    """A multi-column transform that converts text to lower case, removes punctuation, and replaces textual numbers with numeric digits.

    Attributes:
        columns (List[str]): The list of column names to transform.
    """

    columns: List[str]

    def punctuation_case_number_replace(self, text):
        """Replaces punctuation, lowers case, and converts textual numbers to digits.

        Args:
            text (str): The text to transform.

        Returns:
            str: The transformed text, or None if the input is None.
        """
        if text:
            text = re.sub(r"[^\w\s\']", "", text).replace('"', "").replace("'", "").lower().strip()
            # replace words for numbers with numbers, a common occurance in LLM outputs
            number_mapping = {
                "zero": 0,
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
                "six": 6,
                "seven": 7,
                "eight": 8,
                "nine": 9,
            }

            for text_num, num in number_mapping.items():
                pattern = rf"\b{text_num}\b"
                text = re.sub(pattern, str(num), text)

        return text

    def _transform(self, text):
        """Applies the punctuation_case_number_replace method to the given text or list of texts.

        Args:
            text (str or list): The input text or list of texts to transform.

        Returns:
            str or list: The transformed text or list of texts.
        """
        if isinstance(text, list):
            return list(map(self.punctuation_case_number_replace, text))
        else:
            return self.punctuation_case_number_replace(text)


def remove_redundancy(text):
    """Removes redundancy from the text.

    The code is from: https://github.com/alvinmingwisc/spatial_reason_vlm/tree/main/eval,
    and included with minimal modifications.

    Args:
        text (str): The text to remove redundancy from.

    Returns:
        str: The text with redundancy removed.
    """
    text = text.replace("**", "")
    text = text.replace(".", "")
    return text


def extract_before_is(input_string):
    """Extracts the part of the string before the first occurrence of 'is'.

    The code is from: https://github.com/alvinmingwisc/spatial_reason_vlm/tree/main/eval,
    and included with minimal modifications.

    Args:
        input_string (str): The string to process.

    Returns:
        str: A new string containing the part before 'is'.
    """
    parts = input_string.split(" is", 1)
    return parts[0].strip()


def extract_answer_from_text_grid(text, question_type):
    """Extracts the answer from the text based on specific patterns, and as a fallback, extracts the first number if no patterns match.

    The code is from: https://github.com/alvinmingwisc/spatial_reason_vlm/tree/main/eval,
    and included with minimal modifications.

    Args:
        text (str): The text containing the model's answer.
        question_type (str): The text containing the question type.

    Returns:
        str or None: The extracted answer, or None if no answer could be extracted.
    """
    number_mapping = {
        "zero": 0,
        "no": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
    }

    animals = ["giraffe", "cat", "dog", "elephant", "rabbit"]
    animal_pattern = rf"\b({'|'.join(animals)})\b"

    if not text:
        return None
    match = re.search(r"\b[A-D]\.\s*(\w+)", text)
    if match:
        return match.group(1)

    question_id = int(re.search("[0-9]", re.search("Q[0-9]", question_type).group()).group())

    if question_id >= 2:  # Assuming Q4 and Q5 require binary answers
        match = re.search(animal_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    else:
        specific_phrases = [
            (r"\bthere are\s*(\d+)\s*blocks\b", 1),
            (r"\bthere appear to be\s*(\d+)\s*blocks\b", 1),
            (r"\bcontains\s*(\d+)\s*blocks\b", 1),
        ]

        for phrase_pattern, group_index in specific_phrases:
            match = re.search(phrase_pattern, text, re.IGNORECASE)
        if match:
            return match.group(group_index)

        found_numbers = []

        for text_num, num in number_mapping.items():
            for match in re.finditer(rf"\b{text_num}\b", text, re.IGNORECASE):
                found_numbers.append((match.start(), num))

        text = re.sub(r"^\n\n\d+\.\s", "", text)

        for match in re.finditer(r"\d+", text):
            found_numbers.append((match.start(), int(match.group(0))))

        if found_numbers:
            found_numbers.sort(key=lambda x: x[0])
            return str(found_numbers[0][1])

    return None


def extract_answer_from_text_map_and_maze(model_output_raw, options, is_valid, match_first=False):
    """Extracts the answer from the text based on known model output patterns.

    The code is from: https://github.com/alvinmingwisc/spatial_reason_vlm/tree/main/eval,
    and included with minimal modifications.

    Args:
        model_output_raw (str): The text containing the model's answer.
        options (str): The list of options.
        is_valid (bool): Whether the input is valid.
        match_first (bool, optional): If True, the first match is used for answer extraction when multiple matches are found. Defaults to False.

    Returns:
        str or None: The extracted answers, or an empty string if no answer could be extracted.
    """
    model_output_parsed_letter = ""
    model_output_parsed = ""

    if not is_valid or not model_output_raw:
        return ""

    if match_first:
        match_index = 0
    else:
        match_index = -1

    model_output_raw = re.sub(r"\bno objects\b", "0 objects", model_output_raw, re.IGNORECASE)
    model_output_raw = re.sub(r"\bnot\b", "no", model_output_raw, re.IGNORECASE)
    model_output_raw = re.sub(r"\bshould be\b", "is", model_output_raw, re.IGNORECASE)

    number_mapping = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
    }

    for k, v in number_mapping.items():
        model_output_raw = re.sub(rf"\b{k}\b", str(v), model_output_raw, re.IGNORECASE)

    options_dict = {x.split(".")[0].strip().lower(): x.split(".")[1].strip().lower() for x in options}

    model_output_parsed_letter = ""
    model_output_parsed = ""

    answers = [v for k, v in options_dict.items()]
    answers_pattern = rf"\b({'|'.join(answers)})\b"

    if "Answer:".lower() in model_output_raw.lower():
        pattern_letter = r"^\**Answer:\**\s+(\w)\. (\w+)"
        matches = re.findall(pattern_letter, model_output_raw, re.IGNORECASE)
        if matches:
            match_option = matches[match_index].lower()
            if match_option in options_dict:
                model_output_parsed_letter = options_dict[match_option]
            else:
                model_output_parsed_letter = match_option

        pattern_phrase = r"Answer:\**\s+([^\n]+)"
        matches = re.findall(pattern_phrase, model_output_raw, re.IGNORECASE)
        if matches:
            model_output_answer_line = matches[match_index]

            answers_match = re.findall(answers_pattern, model_output_answer_line, re.IGNORECASE)

            if answers_match:
                model_output_parsed = answers_match[match_index]
            else:
                letters = [k for k, v in options_dict.items()]
                letters_pattern = rf"\b({'|'.join(letters)})\b"
                letters_pattern_match = re.findall(letters_pattern, model_output_answer_line, re.IGNORECASE)

                if letters_pattern_match:
                    match_option = letters_pattern_match[match_index].lower()
                    model_output_parsed_letter = options_dict[match_option]

    elif "answer is".lower() in model_output_raw.lower():
        pattern_letter = r"answer is:*\s*\**([\w\d]+)[\s:.]*\**"
        matches = re.findall(pattern_letter, model_output_raw, re.IGNORECASE)
        if matches:
            match_option = matches[match_index].lower()
            if match_option in options_dict:
                model_output_parsed_letter = options_dict[match_option]
            else:
                model_output_parsed_letter = match_option

    model_output_answer_line = model_output_raw.splitlines()[0]

    answers = [v for k, v in options_dict.items()]
    answers_pattern = rf"\b({'|'.join(answers)})\b"
    answers_match = re.findall(answers_pattern, model_output_answer_line, re.IGNORECASE)

    if answers_match:
        model_output_parsed = answers_match[match_index]

    return " or ".join(filter(None, [model_output_parsed, model_output_parsed_letter]))


@dataclass
class ExtractAnswer(DFTransformBase):
    """Base class for an answer extractor that is conditioned on the question type.

    Attributes:
        answer_column_name (str): The column name containing the answer text.
        extracted_answer_column_name (str): The column name for the extracted answer.
        question_type_column_name (str): The column name for the question type.
    """

    answer_column_name: str
    extracted_answer_column_name: str
    question_type_column_name: str

    @abstractmethod
    def _parse_answer_function(self, answer_text, question_type):
        """Parses an answer from textual input based on the question type.

        Args:
            answer_text (str): The answer text to parse.
            question_type (str): The question type.

        Returns:
            str or None: The parsed answer, or None if it cannot be parsed.
        """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the dataframe by parsing answers with the provided parse function.

        Args:
            df (pd.DataFrame): The input dataframe containing the columns for answers.

        Returns:
            pd.DataFrame: A new dataframe with the extracted answers.
        """
        df[self.extracted_answer_column_name] = df.apply(
            lambda x: self._parse_answer_function(x[self.answer_column_name], x[self.question_type_column_name]), axis=1
        )
        return df


@dataclass
class ExtractQuestionOptions(DFTransformBase):
    """Extracts the list of options from a prompt.

    Attributes:
        prompt_column_name (str): The column name containing the prompt text.
        extracted_options_column_name (str): The column name for the extracted options.
    """

    prompt_column_name: str
    extracted_options_column_name: str

    def _extract_options_from_text_map(self, prompt):
        """Extracts the multiple-choice options list from the text.

        Args:
            prompt (str): The text containing the prompt.

        Returns:
            list or None: The extracted list of options, or None if not found.
        """
        prompt_lines = prompt.splitlines()
        matches = [i for i, x in enumerate(prompt_lines) if "Available options:" in x]

        if "Yes" in prompt_lines[matches[0] + 1]:
            options = prompt_lines[matches[0] + 1 : matches[0] + 3]
        else:
            options = prompt_lines[matches[0] + 1 : matches[0] + 5]

        return options

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the dataframe by extracting options from the prompt column.

        Args:
            df (pd.DataFrame): The input dataframe with the prompt column.

        Returns:
            pd.DataFrame: A new dataframe with the extracted options.
        """
        df[self.extracted_options_column_name] = df[self.prompt_column_name].apply(self._extract_options_from_text_map)
        return df


@dataclass
class ExtractAnswerGrid(ExtractAnswer):
    """Answer extractor for the GRID benchmark.

    Attributes:
        answer_column_name (str): The column name containing the answer text.
        extracted_answer_column_name (str): The column name for the extracted answer.
        question_type_column_name (str): The column name for the question type.
        mode (str): The mode of the extractor.
    """

    answer_column_name: str
    extracted_answer_column_name: str
    question_type_column_name: str
    mode: str

    @abstractmethod
    def _parse_answer_function(self, answer_text, question_type):
        """Parses an answer from textual input for the GRID benchmark.

        Args:
            answer_text (str): The answer text to parse.
            question_type (str): The question type.

        Returns:
            str or None: The parsed answer for the GRID benchmark.
        """
        return extract_answer_from_text_grid(answer_text, question_type)


@dataclass
class ExtractAnswerSpatialMapAndMaze(DFTransformBase):
    """Answer extractor for the SPATIAL_MAP and MAZE benchmark.

    Attributes:
        answer_column_name (str): The column name containing the answer text.
        extracted_answer_column_name (str): The column name for the extracted answer.
        extracted_options_column_name (str): The column name containing the extracted options.
        match_first (bool): Whether to match the first occurrence (True) or the last occurrence (False).
    """

    answer_column_name: str
    extracted_answer_column_name: str
    extracted_options_column_name: str
    match_first: bool = False

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the dataframe by extracting answers from text map and maze.

        Args:
            df (pd.DataFrame): The input dataframe that includes an answer column,
                extracted options column, and an "is_valid" column.

        Returns:
            pd.DataFrame: A new dataframe with the extracted answers.
        """
        df[self.extracted_answer_column_name] = df.apply(
            lambda x: extract_answer_from_text_map_and_maze(
                x[self.answer_column_name], x[self.extracted_options_column_name], x["is_valid"], self.match_first
            ),
            axis=1,
        )
        return df
