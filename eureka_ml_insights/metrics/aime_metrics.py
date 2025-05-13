import numpy as np
import logging
from eureka_ml_insights.metrics.metrics_base import ClassicMetric


class NumericMatch(ClassicMetric):
    """This metric class checks numerical match between answer_text and target_text within a margin eps"""
    eps = 1e-6

    def __evaluate__(self, answer_text, target_text, is_valid):
        if not is_valid:
            return "none"
        try:
            diff = np.abs(float(target_text) - float(answer_text))
        except (ValueError, TypeError) as e:
            logging.error(f"failed to extract the numeric values")
            logging.error(f"target_text:'{target_text}', answer_text:'{answer_text}', difference error: {str(e)}")
            return "none"
        if diff < self.eps:
            return "correct"
        else:
            return "incorrect"
