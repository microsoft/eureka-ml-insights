import re
from collections import Counter

from .metrics_base import ClassicMetric

"""
Module containing the MaxTokenF1ScoreMetric class, which calculates the maximum token-based
F1 score from model output and ground truth answers.
"""


class MaxTokenF1ScoreMetric(ClassicMetric):
    """A metric for calculating the maximum token-based F1 score across multiple ground truth answers."""

    def tokenize(self, sentence):
        """Tokenize the input sentence into a list of word tokens.

        Args:
            sentence (str): The input sentence to tokenize.

        Returns:
            list: A list of tokens extracted from the sentence.
        """
        return re.findall(r"\b\w+\b", sentence.lower())

    def __evaluate__(self, model_output, ground_truth, is_valid):
        """Compute the maximum F1 score for the model's output across multiple ground truth answers.

        Args:
            model_output (str): The model's output.
            ground_truth (list of str): A list of ground truth answers.
            is_valid (bool): A flag indicating whether the model's output is valid.

        Returns:
            float: The maximum F1 score among the ground truth answers, or 0 if invalid.
        """
        if not is_valid:
            return 0
        model_answer = model_output
        max_f1 = 0
        for ans in ground_truth:
            f1 = self._f1_score(model_answer, ans)
            max_f1 = max(f1, max_f1)
        return max_f1

    def _f1_score(self, model_output, answer):
        """Compute the F1 score between a model output and a single ground truth answer.

        Args:
            model_output (str): The model's output.
            answer (str): A ground truth answer.

        Returns:
            float: The calculated F1 score between the model output and the ground truth answer.
        """
        target_tokens = self.tokenize(answer.lower())
        generated_tokens = self.tokenize(model_output.lower())

        target_counter = Counter(target_tokens)
        generated_counter = Counter(generated_tokens)

        true_positive = sum((target_counter & generated_counter).values())

        total_generated = len(generated_tokens)
        total_target = len(target_tokens)

        precision = true_positive / total_generated if total_generated > 0 else 0
        recall = true_positive / total_target if total_target > 0 else 0

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return f1