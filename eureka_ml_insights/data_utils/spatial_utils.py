import re
from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import pandas as pd

from .transform import DFTransformBase, MultiColumnTransform


@dataclass
class LowerCaseNoPunctuationConvertNumbers(MultiColumnTransform):
    columns: List[str]

    def punctuation_case_number_replace(self, text):
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
        if isinstance(text, list):
            return list(map(self.punctuation_case_number_replace, text))
        else:
            return self.punctuation_case_number_replace(text)


def remove_redundancy(text):
    """
    The code is from: https://github.com/alvinmingwisc/spatial_reason_vlm/tree/main/eval,
    and included with minimal modifications.
    """
    text = text.replace("**", "")
    text = text.replace(".", "")
    return text


def extract_before_is(input_string):
    """
    This function extracts the part of the string before the first occurrence of 'is'.
    The code is from: https://github.com/alvinmingwisc/spatial_reason_vlm/tree/main/eval,
    and included with minimal modifications.

    :param input_string: The string to process.
    :return: A new string containing the part before 'is'.
    """
    # Split the string at the first occurrence of 'is'
    parts = input_string.split(" is", 1)
    # Return the first part
    return parts[0].strip()


def extract_answer_from_text_grid(text, question_type):
    """
    Extracts the answer from the text based on specific patterns,
    and as a fallback, extracts the first number if no patterns match.
    The code is from: https://github.com/alvinmingwisc/spatial_reason_vlm/tree/main/eval,
    and included with minimal modifications.

    Args:
    - text (str): The text containing the model's answer.
    - question_type (str): The text containing the question type.

    Returns:
    - str or None: The extracted answer, or None if no answer could be extracted.
    """
    # Mapping of textual numbers to their numeric equivalents
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

    # First rule: check for the pattern "A. x", "B. x", "C. x", or "D. x" where x is the answer
    if not text:
        return None
    match = re.search(r"\b[A-D]\.\s*(\w+)", text)
    if match:
        return match.group(1)

    question_id = int(re.search("[0-9]", re.search("Q[0-9]", question_type).group()).group())

    if question_id >= 2:  # Assuming Q4 and Q5 require binary answers
        # Check for animals
        match = re.search(animal_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    else:
        # check answer with numbers
        # Create a list to store all found numbers along with their positions

        # specific for claude
        specific_phrases = [
            (r"\bthere are\s*(\d+)\s*blocks\b", 1),
            (r"\bthere appear to be\s*(\d+)\s*blocks\b", 1),
            (r"\bcontains\s*(\d+)\s*blocks\b", 1),
        ]

        for phrase_pattern, group_index in specific_phrases:
            match = re.search(phrase_pattern, text, re.IGNORECASE)
        if match:
            return match.group(group_index)

        # If no specific phrases found, proceed with other checks
        found_numbers = []

        # Check for textual numbers and their positions
        for text_num, num in number_mapping.items():
            for match in re.finditer(rf"\b{text_num}\b", text, re.IGNORECASE):
                found_numbers.append((match.start(), num))

        # Check for digit sequences and their positions, specifically ignoring list markers at the start
        # Exclude numbers following "\n\n" and directly followed by ". "
        text = re.sub(r"^\n\n\d+\.\s", "", text)  # Remove the leading list marker if it exists

        for match in re.finditer(r"\d+", text):
            found_numbers.append((match.start(), int(match.group(0))))

        # Sort found numbers by their positions (smallest position first)
        if found_numbers:
            found_numbers.sort(key=lambda x: x[0])
            # Return the number associated with the earliest position
            return str(found_numbers[0][1])

    return None  # Return None if no numbers are found


def extract_answer_from_text_map_and_maze(model_output_raw, options, is_valid, match_first=False):
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

    model_output_parsed_letter = ""
    model_output_parsed = ""

    if not is_valid or not model_output_raw:
        return "invalid"

    if match_first:
        match_index = 0
    else:
        match_index = -1

    model_output_raw =  re.sub(r"\bno objects\b", "0 objects", model_output_raw, re.IGNORECASE)
    model_output_raw =  re.sub(r"\bnot\b", "no", model_output_raw, re.IGNORECASE)
    model_output_raw =  re.sub(r"\bshould be\b", "is", model_output_raw, re.IGNORECASE)

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
        model_output_raw =  re.sub(rf"\b{k}\b", str(v), model_output_raw, re.IGNORECASE)

    # get dict of options from options string
    options_dict = {x.split(".")[0].strip().lower():x.split(".")[1].strip().lower() for x in options}


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
                model_output_parsed =  answers_match[match_index]
            else:
                letters = [k for k, v in options_dict.items()]
                letters_pattern = rf"\b({'|'.join(letters)})\b"
                letters_pattern_match = re.findall(letters_pattern, model_output_answer_line, re.IGNORECASE)

                if letters_pattern_match:
                    match_option =  letters_pattern_match[match_index].lower()
                    model_output_parsed_letter = options_dict[match_option]

    elif "answer is".lower() in model_output_raw.lower():
        pattern_letter = r'answer is:*\s*\**([\w\d]+)[\s:.]*\**'
    
        # first look for a single letter answer
        matches = re.findall(pattern_letter, model_output_raw, re.IGNORECASE)
        if matches:
            match_option = matches[match_index].lower()
            if match_option in options_dict:
                model_output_parsed_letter = options_dict[match_option]
            else:
                model_output_parsed_letter = match_option

    # next look if any of the options names are present in the first line

    model_output_answer_line = model_output_raw.splitlines()[0]        

    answers = [v for k, v in options_dict.items()]
    answers_pattern = rf"\b({'|'.join(answers)})\b"
    answers_match = re.findall(answers_pattern, model_output_answer_line, re.IGNORECASE)
    
    if answers_match:
        model_output_parsed =  answers_match[match_index]

    return " or ".join(filter(None, [model_output_parsed, model_output_parsed_letter]))


@dataclass
class ExtractAnswer(DFTransformBase):
    """This class is a base class for an answer extractor that is conditioned on the question type."""

    answer_column_name: str
    extracted_answer_column_name: str
    question_type_column_name: str

    @abstractmethod
    def _parse_answer_function(self, answer_text, question_type):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.extracted_answer_column_name] = df.apply(
            lambda x: self._parse_answer_function(x[self.answer_column_name], x[self.question_type_column_name]), axis=1
        )
        return df

@dataclass
class ExtractQuestionOptions(DFTransformBase):
    """This class is for extracting the option list from a prompt."""

    prompt_column_name: str
    extracted_options_column_name: str

    def _extract_options_from_text_map(self, prompt):
        """
        Extracts the multiple-choice options list from the text.

        Args:
        - text (str): The text containing the prompt.

        Returns:
        - str or None: The extracted list of options.
        """

        # get list of options from prompt
        prompt_lines = prompt.splitlines()
        matches = [i for i, x in enumerate(prompt_lines) if "Available options:" in x]

        if "Yes" in prompt_lines[matches[0]+1]:
            options = prompt_lines[matches[0]+1:matches[0]+3]
        else:
            options = prompt_lines[matches[0]+1:matches[0]+5]

        return options

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.extracted_options_column_name] = df[self.prompt_column_name].apply(self._extract_options_from_text_map)
        return df

@dataclass
class ExtractAnswerGrid(ExtractAnswer):
    """This class is an answer extractor for the GRID benchmark."""

    answer_column_name: str
    extracted_answer_column_name: str
    question_type_column_name: str
    mode: str

    @abstractmethod
    def _parse_answer_function(self, answer_text, question_type):
        return extract_answer_from_text_grid(answer_text, question_type)


@dataclass
class ExtractAnswerSpatialMapAndMaze(DFTransformBase):
    """This class is an answer extractor for the SPATIAL_MAP and MAZE benchmark."""

    answer_column_name: str
    extracted_answer_column_name: str
    extracted_options_column_name: str
    match_first: bool = False    

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.extracted_answer_column_name] = df.apply(
            lambda x: extract_answer_from_text_map_and_maze(x[self.answer_column_name], x[self.extracted_options_column_name], x["is_valid"], self.match_first), axis=1
        )
        return df
