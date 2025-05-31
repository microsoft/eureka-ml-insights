import logging
from eureka_ml_insights.metrics.metrics_base import ClassicMetric
import json
import ast
import logging

class DictMatch(ClassicMetric):
    """This metric class checks if two dictionary strings represent the same dictionary."""

    def __evaluate__(self, answer_text, target_text, is_valid):
        if not is_valid:
            return "none"

        try:
            answer_dict = self._parse_to_dict(answer_text)
            target_dict = self._parse_to_dict(target_text)
        except ValueError as e:
            logging.error(f"Failed to parse one of the inputs as a dict.")
            logging.error(f"target_text:'{target_text}', answer_text:'{answer_text}', error: {str(e)}")
            return "none"

        if answer_dict == target_dict:
            return "correct"
        else:
            return "incorrect"

    def _parse_to_dict(self, s):
        """Try to parse a string to a Python dict using JSON or Python literal syntax."""
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                result = ast.literal_eval(s)
                if isinstance(result, dict):
                    return result
            except Exception:
                pass
        raise ValueError(f"Invalid dictionary string: {s}")
