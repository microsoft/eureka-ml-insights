"""This module provides the IFEvalMetric class for evaluating if a response follows instructions.

The IFEvalMetric extends the CompositeMetric class to measure compliance with instructions
using both strict and loose evaluations.
"""

from eureka_ml_insights.metrics.ifeval_instruction_utils import (
    test_instruction_following_loose,
    test_instruction_following_strict,
)
from eureka_ml_insights.metrics.metrics_base import CompositeMetric


class IFEvalMetric(CompositeMetric):
    """Composite metric for evaluating if a response follows instructions.

    This class calculates both strict and loose evaluation scores based on the response's
    adherence to the instructions by leveraging the CompositeMetric base class.
    """

    def __init__(self):
        """Initializes the IFEvalMetric instance."""
        super().__init__()

    def __evaluate__(self, row):
        """Evaluates the row data and returns the instruction following metric results.

        Args:
            row (dict): A dictionary containing various keys such as 'instruction_id_list' and
                'is_valid' that describe the prompt-response pair and other metadata.

        Returns:
            dict: A dictionary containing the evaluation results, which includes metrics on how
            well the response follows instructions.
        """
        tier0_instructions = []
        for index, instruction_id in enumerate(row["instruction_id_list"]):
            tier0_instructions.append(instruction_id.split(":")[0])

        if not row["is_valid"]:
            return {
                "strict_follow_all_instructions": 0,
                "strict_follow_instruction_list": [0 for inst in row["instruction_id_list"]],
                "strict_instruction_list_len": 0,
                "strict_follow_instruction_list_sum": 0,
                "tier0_instructions": tier0_instructions,
                "loose_follow_all_instructions": 0,
                "loose_follow_instruction_list": [0 for inst in row["instruction_id_list"]],
                "loose_instruction_list_len": 0,
                "loose_follow_instruction_list_sum": 0,
                "is_valid": False,
            }
        return self.evaluate_instruction_following(row)

    def evaluate_instruction_following(self, row, include_loose_evaluation=True):
        """Evaluates instruction following for a given row.

        Args:
            row (dict): The dictionary containing 'prompt', 'response', 'instruction_id_list',
                'kwargs', and other relevant data.
            include_loose_evaluation (bool): Whether to include loose evaluation in addition to
                strict evaluation. Defaults to True.

        Returns:
            dict: A dictionary combining the results from strict evaluation and, if enabled, loose
            evaluation.
        """
        results_strict = test_instruction_following_strict(
            row["prompt"], row["response"], row["instruction_id_list"], row["kwargs"]
        )
        results_loose = {}
        if include_loose_evaluation:
            results_loose = test_instruction_following_loose(
                row["prompt"], row["response"], row["instruction_id_list"], row["kwargs"]
            )
        return {**results_strict, **results_loose}
