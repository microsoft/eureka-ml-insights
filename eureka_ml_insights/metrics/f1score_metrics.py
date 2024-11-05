import re
from collections import Counter

from .metrics_base import ClassicMetric


class MaxTokenF1ScoreMetric(ClassicMetric):
    def validate_data(self, data):
        """This method checks if the data has the required fields."""
        assert "model_output" in data.columns, "Data does not have 'model_output' field."
        assert "ground_truth" in data.columns, "Data does not have 'ground_truth' field."
        return True

    def tokenize(self, sentence):
        return re.findall(r"\b\w+\b", sentence.lower())

    # Function to compute F1 score between two responses
    def evaluate_f1(self, model_output, ground_truth):
        model_answer = model_output
        max_f1 = 0
        for ans in ground_truth:
            f1 = self._f1_score(model_answer, ans)
            max_f1 = max(f1, max_f1)
        return max_f1

    def _f1_score(self, model_output, answer):
        # Tokenize both responses
        target_tokens = self.tokenize(answer.lower())
        generated_tokens = self.tokenize(model_output.lower())

        # Count tokens
        target_counter = Counter(target_tokens)
        generated_counter = Counter(generated_tokens)

        # Calculate true positives
        true_positive = sum((target_counter & generated_counter).values())

        # Total tokens in generated and target responses
        total_generated = len(generated_tokens)
        total_target = len(target_tokens)

        # Precision and recall calculations
        precision = true_positive / total_generated if total_generated > 0 else 0
        recall = true_positive / total_target if total_target > 0 else 0

        # Calculate F1 score
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return f1

    def evaluate(self, data):
        self.validate_data(data)
        data[self.__class__.__name__ + "_result"] = data.apply(
            lambda x: self.evaluate_f1(x["model_output"], x["ground_truth"]),
            axis=1,
        )
        return data
