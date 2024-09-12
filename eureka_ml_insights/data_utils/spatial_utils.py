import re
from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import pandas as pd

from eureka_ml_insights.data_utils.transform import (
    DFTransformBase,
    MultiColumnTransform,
)


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


def extract_answer_from_text_map(text, question_type, model_name):
    """
    Extracts the answer from the text based on specific patterns,
    and as a fallback, extracts the first number if no patterns match.
    The code is from: https://github.com/alvinmingwisc/spatial_reason_vlm/tree/main/eval,
    and included with minimal modifications.

    Args:
    - text (str): The text containing the model's answer.
    - question_type (str): The text containing the question type.
    - model_name (str): The model name.

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

    dirs = ["southeast", "northeast", "northwest", "southwest"]
    dir_pattern = rf"\b({'|'.join(dirs)})\b"

    if text is None:
        return None

    question_id = int(re.search("[0-9]", re.search("Q[0-9]", question_type).group()).group())

    if question_id == 0:
        direction_match = re.search(r"\b[A-D]\.\s*(" + "|".join(dirs) + r")\b", text, re.IGNORECASE)
        if direction_match:
            return direction_match.group(1).lower()

        match = re.search(dir_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    elif question_id == 1:
        match = re.search(
            rf"^([\w\s\'\']+?)\s+is\s+(?:located\s+|in\s+the\s+|located\s+to\s+the\s+)({dir_pattern})",
            text,
            re.IGNORECASE,
        )

        if match:
            string = match.group(1)
            return string

        match = re.search(r"\b[A-D]\.\s*(.*)", text)  # problem with extracting .

        if match:
            string = match.group(1)
            string = remove_redundancy(string)
            string = extract_before_is(string)
            return string

        match = re.search(r"\b([ABCD][.,]|[(][abcdABCD][)])\s*(.*?)(?=\sis\b|\.|,|<|$)", text)
        if match:
            answer = match.group(1).strip()
            # Remove trailing punctuation if any
            answer = re.sub(r"[\.,\?!<]+$", "", answer)
            return answer

        match = re.search(
            rf"Therefore, the object in the {dir_pattern} of [\w\s\'\']+ is ([\w\s\'\']+)", text, re.IGNORECASE
        )
        if match:
            string = match.group(2)
            return string

        if "claude" in model_name.lower():
            match = re.search(rf"^([\w\s\'\']+?)\s+is\s+(to\s+the\s+)({dir_pattern})", text, re.IGNORECASE)
            if match:
                string = match.group(1)
                return string

        if "gemini" in model_name.lower():
            patterns = [
                rf"\*\*Concise Answer:\*\*\n([\w\s\'\']+?)\s+is\s+(?:located\s+|in\s+the\s+|in\s+|located\s+to\s+the\s+)({dir_pattern})",
                rf"\*\*Answer:\*\*\s+([\w\s\'\']+?)\s+is\s+in\s+the\s+({dir_pattern})\s+of\s+([\w\s\'\']+)",
                r"\*\*Answer:\*\*\n([\w\s\'\']+)",
                r"\*\*Answer\*\*:\s+([\w\s\'\']+)",
                r"\*\*Answer:\*\*\s+([\w\s\'\']+)",
            ]

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1)

        if "gpt-4o" in model_name.lower() or "gpt4o" in model_name.lower():
            match = re.search(
                rf"Concise Answer:\s+([\w\s\'\']+?)\s+is\s+(?:located\s+|in\s+the\s+|in\s+|located\s+to\s+the\s+)({dir_pattern})",
                text,
                re.IGNORECASE,
            )
            if match:
                string = match.group(1)
                return string

        # If no match, check for an answer following "is", with specific end markers defined
        match = re.search(r"\bis\b\s+(.*?)(?=\.|,|<|$)", text)
        if match:
            answer = match.group(1).strip()
            # Remove trailing punctuation if any
            answer = re.sub(r"[\.,\?!<]+$", "", answer)
            return answer

        return None  # Return None if no match is found

    elif question_id == 2:
        match = re.search(r"\b[A-D]\.\s*(\d+)", text)  # match number only
        if match:
            return match.group(1)
        # Create a list to store all found numbers along with their positions
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
        return None

    else:
        raise ValueError(f"Question ID {question_id} is not supported.")

    return None  # Return None if no numbers are found


def extract_answer_from_text_maze(text, question_type):
    """
    Extracts the answer from the text based on specific patterns including handling
    variations with 'no right turn', 'no right turns', answers separated by new lines,
    path descriptions, textual numbers, and as a fallback, extracts the first number if no patterns match.

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

    question_id = int(re.search("[0-9]", re.search("Q[0-9]", question_type).group()).group())

    if question_id in [3, 4]:  # Assuming Q4 and Q5 require binary answers
        yes_patterns = [
            r"\byes\b",
            r"the answer is yes",
            r"\"yes\"",
            r"\'yes\'",
            r"\'Yes\'",
            r"is the shortest path",
            r"A\. Yes",
        ]
        no_patterns = [r"\bno\b", r"the answer is no", r"\"no\"", r"\'no\'", r"\'No\'", r"\bnot\b", r"B\. No"]

        if not text:
            return None

        # Check for "Yes" answers
        for pattern in yes_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "Yes"

        # Check for "No" answers
        for pattern in no_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "No"

    else:
        # check question ask for number
        # First rule: check for the pattern "A. x", "B. x", "C. x", or "D. x" where x is the answer
        if not text:
            return None
        match = re.search(r"\b[A-D]\.\s*(\d+)", text)  # match number only
        if match:
            return match.group(1)

        patterns = {
            0: [  # For right turns
                # r'\b(\d+)\s+right turn(s)?',
                r"\bThere are\s*(\d+)\s*right turns\b",  # for proprietary
                r"\bThere is\s*(\d+)\s*right turn\b",
                r"answer is\s+(\d+)",
                r"answer is:\s*\n*\s*(\d+)",
                r"from S to E is\s+(\d+)",
                r"Answer:\*\*\s*(\d+)\b",
            ],
            1: [  # For left turns
                r"\b(\d+)\s+left turn(s)?",
                r"answer is\s+(\d+)",
                r"answer is:\s*\n*\s*(\d+)",
                r"from S to E is\s+(\d+)",
            ],
            2: [  # For total turns
                r"\bThere are\s*(\d+)\s*total turns\b",  # for proprietary
                r"\bThere are\s*(\d+)\s*turns\b",
                r"There is\s*(\d+)\s*turn\b",
                r"There is\s*(\d+)\s*total turn\b",
                r"answer is\s+(\d+)",
                r"answer is:\s*\n*\s*(\d+)",
                r"from S to E is\s+(\d+)",
                r"\btotal of\s+(\d+)\s+turn(s)?",
                r"Answer:\*\*\s*(\d+)\b",
                # r'(\d+)\s+total turn(s)?',
                # r'There are (\d+)',
                # r'\b(\d+)\s+turn(s)?',  # Matches "8 turns" broadly; consider specificity vs. overlap
            ],
        }

        for pattern in patterns[question_id]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)  # Return the first matching group as integer

        # Check for textual number patterns
        for text_num, num in number_mapping.items():
            pattern = rf"\b{text_num}\s*(right|left|total)?\s*turn(s)?\b"
            if re.search(pattern, text, re.IGNORECASE):
                return str(num)

        # If no specific pattern matches, try to extract the first number in the text
        fallback_match = re.search(r"\d+", text)
        if fallback_match:
            return fallback_match.group(0)  # Return the matched number

    return None  # Return None if no number or textual number is found at all


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
class ExtractAnswerSpatialMap(ExtractAnswer):
    """This class is an answer extractor for the SPATIAL_MAP benchmark."""

    answer_column_name: str
    extracted_answer_column_name: str
    question_type_column_name: str
    model_name: str

    @abstractmethod
    def _parse_answer_function(self, answer_text, question_type):
        return extract_answer_from_text_map(answer_text, question_type, self.model_name)


@dataclass
class ExtractAnswerMaze(ExtractAnswer):
    """This class is an answer extractor for the MAZE benchmark."""

    answer_column_name: str
    extracted_answer_column_name: str
    question_type_column_name: str

    @abstractmethod
    def _parse_answer_function(self, answer_text, question_type):
        return extract_answer_from_text_maze(answer_text, question_type)
