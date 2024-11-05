import unittest
from eureka_ml_insights.metrics import MaxTokenF1ScoreMetric

class MaxTokenF1ScoreMetricTest(unittest.TestCase):
    def setUp(self):
        self.metric = MaxTokenF1ScoreMetric()

        # Sample test data
        self.values = [
            {
                "model_output": "The quick brown fox",
                "ground_truth": ["The quick brown fox", "A quick brown fox jumps"],
                "expected_f1": 1.0  # Exact match
            },
            {
                "model_output": "A fast brown dog",
                "ground_truth": ["The quick brown fox", "Brown dog, fast fox", "Dogs are brown."],
                "expected_f1": 0.75  # Partial match
            },
            {
                "model_output": "Jumping over the lazy dog",
                "ground_truth": ["The quick brown fox jumps", "A lazy dog lies here", "The jumping dog isn't lazy"],
                "expected_f1": 0.73  # Partial match
            },
            {
                "model_output": "Completely different text",
                "ground_truth": ["An unrelated sentence", "Another distinct phrase", "Something else as well"],
                "expected_f1": 0.0  # No match
            },
        ]

    def test_evaluate(self):
        for i, val in enumerate(self.values):
            result = self.metric.evaluate_f1(val["model_output"], val["ground_truth"])
            self.assertAlmostEqual(result, val["expected_f1"], places=2)

if __name__ == "__main__":
    unittest.main()
