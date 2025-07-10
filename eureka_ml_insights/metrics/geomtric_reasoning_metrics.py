import ast

from .spatial_and_layout_metrics import SpatialAndLayoutReasoningMetric


class GeoMCQMetric(SpatialAndLayoutReasoningMetric):
    """This class is a metric that requires a correct prediction to be only one of the valid multiple choice answers."""

    def __init__(self):
        super().__init__()

    def __evaluate__(self, answer_text, target_text, target_options, is_valid):
        if not is_valid:
            return "none"

        target_options = ast.literal_eval(target_options)

        return super().__evaluate__(answer_text, target_text, target_options, is_valid)
