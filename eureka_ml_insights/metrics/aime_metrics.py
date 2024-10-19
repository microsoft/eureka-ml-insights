from tqdm.auto import tqdm

from eureka_ml_insights.metrics.metrics_base import ClassicMetric


class AIME_ExactMatch(ClassicMetric):
    """This class checks for an exact match."""

    def __evaluate__(self, answer_text, target_text, is_valid):
        if not is_valid:
            return 0
        if target_text == answer_text:
            return 1
        else:
            return 0