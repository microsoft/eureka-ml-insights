"""
This module provides a metric class for checking the numerical match between
two string representations of numeric values.
"""

import numpy as np
import logging
from eureka_ml_insights.metrics.metrics_base import ClassicMetric


class NumericMatch(ClassicMetric):
    """NumericMatch class.

    This metric class checks the numerical match between answer_text and
    target_text within a margin of eps.

    Attributes:
        eps (float): Tolerance for the difference between numerical values.
    """

    eps = 1e-6

    def __evaluate__(self, answer_text, target_text, is_valid):
        """Evaluate the match between answer_text and target_text.

        Args:
            answer_text (str): The predicted numeric value as a string.
            target_text (str): The actual numeric value as a string.
            is_valid (bool): Indicates whether the answer is valid or not.

        Returns:
            str: "correct" if the difference between the numeric values is within eps,
            otherwise "incorrect". Returns "none" if the numeric values cannot be parsed
            or if is_valid is False.
        """
        if not is_valid:
            return "none"
        try:
            diff = np.abs(float(target_text) - float(answer_text))
        except (ValueError, TypeError) as e:
            logging.error(f"failed to extract the numeric values")
            logging.error(f"target_text:'{target_text}', answer_text:'{answer_text}', error: {str(e)}")
            return "none"
        if diff < self.eps:
            return "correct"
        else:
            return "incorrect"