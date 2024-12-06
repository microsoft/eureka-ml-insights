import ast

from .metrics_base import ExactMatch
import xml.etree.ElementTree as ET


class NPHardMetric(ExactMatch):
    """This class is a metric that requires a correct prediction to be only one of the valid multiple choice answers."""

    def __init__(self):
        super().__init__()

    def parse_xml_to_dict(self, xml_string):
        # Parse the XML string
        root = ET.fromstring(xml_string)

        # Find the 'final_answer' tag
        final_answer_element = root.find('final_answer')

        # Find the 'reasoning' tag
        reasoning_element = root.find('reasoning')

        return final_answer_element, reasoning_element

    def __evaluate__(self, answer_text, target_text, is_valid):
        if not is_valid:
            return "none"

        final_answer_element, reasoning_element = self.parse_xml_to_dict(answer_text)
        tour_distance = ast.literal_eval(final_answer_element.text)['TotalDistance']

        return super().__evaluate__(tour_distance, target_text, is_valid)
