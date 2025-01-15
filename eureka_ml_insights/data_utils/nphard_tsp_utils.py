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

def convert_city_path(path):
    """
    Convert a path in the format 'City1->City6->City7->...' into '1->6->7->...'.

    Args:
        path (str): The input path in the format 'CityX->CityY->...'.

    Returns:
        str: The converted path in numerical format.
    """
    # Split the input path into individual cities
    city_list = path.split("->")

    # Extract the numeric part from each city and join them with '->'
    numeric_path = "->".join(city.replace("City", "") for city in city_list)

    return numeric_path

def parse_xml_to_dict(xml_string):
    # Parse the XML string
    # breakpoint()
    root = ET.fromstring(xml_string)
    final_answer_element = root.find('final_answer')

    tour_string = ast.literal_eval(final_answer_element.text)['Path']

    if 'City' in tour_string:
        tour_string = convert_city_path(tour_string)

    tour = list(map(int, tour_string.split('->')))

    tour_string = ','.join(map(str, tour))

    return tour_string
