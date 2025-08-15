import re, json, ast
from dataclasses import dataclass

import pandas as pd

from .transform import DFTransformBase


@dataclass
class BFCLExtractAnswer(DFTransformBase):
    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.model_answer_column] = df[self.model_output_column].apply(self.parse_output_answer)
        return df

    @staticmethod
    def parse_output_answer(response):
        """
        Parse the input string to extract answer of a given AIME question.
        Allows and ignores optional <think>...</think> blocks in the response.

        Parameters:
            response (str): Input string containing answer X in the form of "Final Answer: X".
        Returns:
            numerical_value (float or str): A numeric value or JSON string representing the model's answer.
        """
        return extract_last_json_dict(response)
        # Remove <think>...</think> blocks
        response_cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        try:
            json.loads(response_cleaned)
            return response_cleaned  # valid JSON
        except json.JSONDecodeError:
            try:
                result = ast.literal_eval(response_cleaned)
                if isinstance(result, dict):
                    return json.dumps(result)
            except Exception:
                return None

        return None

@dataclass
class BFCLMultiturnExtractAnswer(DFTransformBase):
    model_output_column: str
    model_answer_column: str
    num_iter: int

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.model_answer_column] = df[self.model_output_column].apply(self.parse_output_answer)
        df[self.model_answer_column] = df[self.model_output_column].apply(
            lambda x: self.parse_output_answer(x, num_iter=self.num_iter)
            )
        return df

    @staticmethod
    def parse_output_answer(response,num_iter=3):
        """
        Parse the input string to extract answer of a given AIME question.
        Allows and ignores optional <think>...</think> blocks in the response.

        Parameters:
            response (str): Input string containing answer X in the form of "Final Answer: X".
        Returns:
            The output is a list[list[list[str]]]
            The first index is the turn;
            The second index is the step;
            The third index is the list of func calls extracted from the message.
        """
        print(type(response))
        print(repr(response))

        #all_messages = json.loads(response)
        all_messages = response

        extracted_answers = extract_nested_function_calls(all_messages)
        formated_answers = format_function_calls(extracted_answers,num_iter=num_iter)
        return formated_answers

        # Remove <think>...</think> blocks
        response_cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        try:
            json.loads(response_cleaned)
            return response_cleaned  # valid JSON
        except json.JSONDecodeError:
            try:
                result = ast.literal_eval(response_cleaned)
                if isinstance(result, dict):
                    return json.dumps(result)
            except Exception:
                return None

        return None
    
import json
import re

import re
from typing import List

import re
from typing import List, Dict

def extract_nested_function_calls(messages: List[Dict[str, str]]) -> List[List[List[str]]]:
    """
    Extract all assistant function calls in nested list structure:
    List[message] -> List[group] -> List[function calls without brackets].

    Returns:
        List[List[List[str]]]: e.g., [[["touch(...)"]], [["load(...)", "analyze()"]]]
    """
    pattern = re.compile(r"\[([a-zA-Z_][a-zA-Z0-9_]*\([^]]*\))\]")
    result = []

    for msg in messages:
        if msg.get("role") == "assistant":
            calls = pattern.findall(msg.get("content", ""))
            if calls:
                result.append([calls])  # Wrap in two levels of lists
            else:
                result.append([])  # No function calls

    return result


def format_function_calls(extracted_answer: List[List[List[str]]], num_iter=3 ) -> List[List[List[str]]]:
    """Group every x items into one list without adding an extra nesting level."""
    grouped = []
    for i in range(0, len(extracted_answer), num_iter):
        chunk = []
        for sublist in extracted_answer[i:i+num_iter]:
            chunk.extend(sublist)  # append elements directly instead of nesting
        grouped.append(chunk)
    return grouped

def extract_clean_function_calls(messages: List[dict]) -> List[str]:
    """
    Extract and clean all function calls from assistant messages.
    Returns each function call without surrounding square brackets.

    Args:
        messages (List[dict]): A list of messages with "role" and "content".

    Returns:
        List[str]: A flat list of function call strings without brackets.
    """
    pattern = re.compile(r"\[([a-zA-Z_][a-zA-Z0-9_]*\([^]]*\))\]")
    result = []

    for msg in messages:
        if msg.get("role") == "assistant":
            matches = pattern.findall(msg.get("content", ""))
            result.extend(matches)

    return result


def extract_last_json_dict(text):
    """
    Extract the last complete outermost JSON dictionary from the given text.
    Handles nested dictionaries and ignores other non-JSON content.
    
    Parameters:
        text (str): Input text that may contain JSON dictionary.
    Returns:
        str or None: The last valid JSON dict string, or None if not found.
    """
    # Remove <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    stack = []
    json_candidates = []
    start_idx = None

    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start_idx = i
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    candidate = text[start_idx:i+1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            json_candidates.append(candidate)
                    except Exception:
                        pass
                    start_idx = None

    return json_candidates[-1] if json_candidates else None
