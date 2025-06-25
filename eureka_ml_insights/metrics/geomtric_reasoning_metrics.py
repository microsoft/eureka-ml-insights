"""Module containing the GeoMCQMetric class for evaluating multiple-choice questions.

GeoMCQMetric is derived from SpatialAndLayoutReasoningMetric and enforces that a 
correct prediction must be one of the valid multiple-choice options.
"""

import ast

from .spatial_and_layout_metrics import SpatialAndLayoutReasoningMetric


class GeoMCQMetric(SpatialAndLayoutReasoningMetric):
    """
    A metric that requires a correct prediction to be only one of the valid multiple-choice answers.
    """

    def __init__(self):
        """
        Initializes the GeoMCQMetric class and calls the parent class constructor.
        """
        super().__init__()

    def __evaluate__(self, answer_text, target_text, target_options, is_valid):
        """
        Evaluates whether the predicted answer matches the target answer within valid multiple-choice options.

        Args:
            answer_text (str): The predicted answer text.
            target_text (str): The ground truth answer text.
            target_options (str): The multiple-choice options as a string representation (e.g., a list).
            is_valid (bool): Whether the context for evaluation is valid.

        Returns:
            str: The evaluated result. Returns "none" if not valid; otherwise, calls the superclass evaluation.
        """
        if not is_valid:
            return "none"

        target_options = ast.literal_eval(target_options)

        return super().__evaluate__(answer_text, target_text, target_options, is_valid)