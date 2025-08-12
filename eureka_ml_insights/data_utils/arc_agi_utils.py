import re
from dataclasses import dataclass

import pandas as pd

from .transform import DFTransformBase


@dataclass
class ARCAGI_ExtractAnswer(DFTransformBase):
    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.model_answer_column] = df[self.model_output_column].apply(self.parse_output_answer)
        return df

    @staticmethod
    def parse_output_answer(response):
        """
        Parse the input string to extract answer of a given ARCAGI question.
        Parameters:
            response (str): Input string containing answer X in the form of "<output>final answer string</output>".
        Returns: 
            answer (str): The final answer string with leading and training spaces stripped.
        """
        answer = ""

        if response is None:
            return ""
        elif response.find("<output>") == -1 or response.find("</output>") == -1:
            return ""

        start_index = response.find("<output>") + len("<output>")
        end_index = response.find("</output>")

        answer = response[start_index:end_index].strip()

        return answer


@dataclass
class ARCAGI_CleanCOTAnswer(DFTransformBase):
    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.model_answer_column] = df[self.model_output_column].apply(self.parse_output_answer)
        return df

    @staticmethod
    def parse_output_answer(response):
        """
        Replace None responses with an empty string
        Parameters:
            response (str): Possibly None Response string
        Returns: 
            answer (str): Response string with None replaced by blank string
        """
        if response is None:
            return ""
        
        start_index = response.find("</think>") + len("</think>")
        if start_index == -1:
            return response
        
        response = response[start_index:]

        return response
