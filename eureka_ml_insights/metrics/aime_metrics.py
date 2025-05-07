import numpy as np

from eureka_ml_insights.metrics.metrics_base import ClassicMetric


class NumericMatch(ClassicMetric):
    """This class checks for a numeric match."""

    eps = 1e-6

    def __evaluate__(self, answer_text, target_text, is_valid):
        if not is_valid:
            return "none"
        try:
            diff = np.abs(float(target_text) - float(answer_text))
        except:
            return "none"
        if diff < self.eps:
            return "correct"
        else:
            return "incorrect"
