import unittest

from eureka_ml_insights.metrics import SpatialAndLayoutReasoningMetric


class LayoutReasoningMetricTest(unittest.TestCase):
    def setUp(self):
        self.metric = SpatialAndLayoutReasoningMetric()

        self.values = [
            {
                "target_options": ["Southwest", "Northwest", "North", "Northeast"],
                "ground_truth": "Northeast",
                "model_output": "Northeast",
                "is_valid": "true",
            },
            {
                "target_options": ["Southwest", "Northwest", "North", "Northeast"],
                "ground_truth": "Northeast",
                "model_output": "The answer is North.",
                "is_valid": "true",
            },  # tricky substring case where need to handle that North and Northwest shouldn't match
            {
                "target_options": ["Southwest", "Northwest", "North", "Northeast"],
                "ground_truth": "Northeast",
                "model_output": "[Southwest, Northwest, North, Northeast]",
                "is_valid": "true",
            },
            {
                "target_options": [
                    "Jasmine's Jewellery",
                    "Tapas Temptations",
                    "Tiger's Tapestries",
                    "Police Supply Store",
                ],
                "ground_truth": "Police Supply Store",
                "model_output": "The answer is Police Supply Store.  I hope that helps.",
                "is_valid": "true",
            },
            {
                "target_options": [
                    "Jasmine's Jewellery",
                    "Tapas Temptations",
                    "Tiger's Tapestries",
                    "Police Supply Store",
                ],
                "ground_truth": "Police Supply Store",
                "model_output": "The answer is Jasmine's Jewellery, I think",
                "is_valid": "true",
            },
            {
                "target_options": [
                    "Jasmine's Jewellery",
                    "Tapas Temptations",
                    "Tiger's Tapestries",
                    "Police Supply Store",
                ],
                "ground_truth": "Police Supply Store",
                "model_output": "Tiger's Tapestries, Police Supply Store",
                "is_valid": "true",
            },
            {"target_options": ["4", "3", "5", "0"], "ground_truth": "0", "model_output": "0", "is_valid": "true"},
            {"target_options": ["4", "3", "5", "0"], "ground_truth": "0", "model_output": "4", "is_valid": "true"},
            {"target_options": ["4", "3", "5", "0"], "ground_truth": "0", "model_output": "10", "is_valid": "true"},
        ]

    def test_layout_reasoning_metric(self):
        expected_results = [
            "correct",
            "incorrect",
            "none",
            "correct",
            "incorrect",
            "none",
            "correct",
            "incorrect",
            "none",
        ]

        results = []
        for i, val in enumerate(self.values):
            result = self.metric.__evaluate__(
                val["model_output"], val["ground_truth"], val["target_options"], val["is_valid"]
            )
            results.append(result)

        self.assertListEqual(expected_results, results)


if __name__ == "__main__":
    unittest.main()
