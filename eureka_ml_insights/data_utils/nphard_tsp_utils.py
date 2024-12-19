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
        df[self.model_answer_column] = df[self.model_output_column].apply(parse_xml_to_dict)
        return df


def parse_xml_to_dict(xml_string):
    # Parse the XML string
    root = ET.fromstring(xml_string)
    final_answer_element = root.find('final_answer')

    tour_string = ast.literal_eval(final_answer_element.text)['Path']
    tour = list(map(int, tour_string.split('->')))

    tour_string = ','.join(map(str, tour))

    return tour_string

    # return tour



##########################

@dataclass
class NPHARDTSPExtractAnswer1(DFTransformBase):
    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.model_answer_column] = df[self.model_output_column].apply(parse_xml_to_dict1)
        return df

# def parse_output_answer(response):

def parse_xml_to_dict1(xml_string):
    # Parse the XML string
    root = ET.fromstring(xml_string)
    final_answer_element = root.find('final_answer')

    tour_string = ast.literal_eval(final_answer_element.text)['Path']
    tour = list(map(int, tour_string.split('->')))
    # tour_string = ''.join(map(str, tour))
    tour_string = ','.join(map(str, tour))

    return tour_string