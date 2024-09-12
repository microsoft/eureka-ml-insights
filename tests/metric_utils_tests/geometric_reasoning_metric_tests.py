import unittest

from eureka_ml_insights.metrics import GeoMCQMetric


class GeometricReasoningMetricTest(unittest.TestCase):
    def setUp(self):
        self.metric = GeoMCQMetric()

        self.values = [
            {
                "target_options": "['(brown triangle, orange rectangle)', '(orange rectangle, brown triangle)']",
                "ground_truth": "(orange rectangle, brown triangle)",
                "model_output": "(brown triangle, orange rectangle)",
                "is_valid": "true",
            },
            {
                "target_options": "['(purple triangle, yellow rectangle, blue triangle)', \
                '(blue triangle, purple triangle, yellow rectangle)', \
                '(yellow rectangle, purple triangle, blue triangle)', \
                '(blue triangle, yellow rectangle, purple triangle)', \
                '(yellow rectangle, blue triangle, purple triangle)', \
                '(purple triangle, blue triangle, yellow rectangle)']",
                "ground_truth": "(purple triangle, blue triangle, yellow rectangle)",
                "model_output": "(purple triangle, blue triangle, yellow rectangle)",
                "is_valid": "true",
            },
            {
                "target_options": "['(A, C, B)', '(A, B, C)', '(B, A, C)', '(C, A, B)', '(B, C, A)', '(C, B, A)']",
                "ground_truth": "(A, C, B)",
                "model_output": "(A, C, B)",
                "is_valid": "true",
            },
            {
                "target_options": "['(A, C, B)', '(A, B, C)', '(B, A, C)', '(C, A, B)', '(B, C, A)', '(C, B, A)']",
                "ground_truth": "(A, C, B)",
                "model_output": "(A, B, C)",
                "is_valid": "true",
            },
            {
                "target_options": "['(76, 87, 81, 62, 47)', \
                    '(81, 76, 47, 87, 62)', \
                    '(47, 76, 81, 62, 87)', \
                    '(87, 47, 62, 76, 81)', \
                    '(81, 87, 76, 62, 47)', \
                    '(76, 81, 87, 62, 47)', \
                    '(87, 47, 81, 76, 62)', \
                    '(76, 62, 81, 47, 87)', \
                    '(62, 87, 76, 47, 81)', \
                    '(62, 76, 87, 81, 47)', \
                    '(47, 81, 62, 87, 76)', \
                    '(81, 76, 62, 87, 47)', \
                    '(81, 87, 76, 47, 62)', \
                    '(81, 62, 87, 76, 47)', \
                    '(62, 76, 81, 87, 47)', \
                    '(81, 47, 76, 87, 62)', \
                    '(62, 87, 47, 76, 81)', \
                    '(87, 47, 76, 62, 81)', \
                    '(47, 62, 76, 81, 87)', \
                    '(62, 81, 87, 47, 76)', \
                    '(87, 47, 62, 81, 76)', \
                    '(47, 62, 81, 87, 76)', \
                    '(81, 87, 62, 76, 47)', \
                    '(81, 87, 47, 76, 62)', \
                    '(62, 76, 47, 81, 87)', \
                    '(47, 76, 87, 62, 81)', \
                    '(62, 87, 47, 81, 76)', \
                    '(87, 47, 76, 81, 62)', \
                    '(87, 62, 81, 47, 76)', \
                    '(47, 62, 76, 87, 81)', \
                    '(62, 87, 81, 47, 76)', \
                    '(76, 81, 62, 47, 87)', \
                    '(47, 87, 76, 62, 81)', \
                    '(87, 76, 81, 47, 62)', \
                    '(81, 76, 87, 62, 47)', \
                    '(62, 81, 76, 87, 47)', \
                    '(62, 47, 81, 76, 87)', \
                    '(47, 87, 76, 81, 62)', \
                    '(62, 81, 47, 76, 87)', \
                    '(47, 87, 62, 76, 81)', \
                    '(47, 62, 81, 76, 87)']",
                "ground_truth": "(62, 81, 76, 87, 47)",
                "model_output": "(62, 81, 76, 87, 47)",
                "is_valid": "true",
            },
            {
                "target_options": "['(brown rectangle, red triangle, magenta triangle, cyan triangle)', \
                '(magenta triangle, brown rectangle, red triangle, cyan triangle)', \
                '(cyan triangle, magenta triangle, brown rectangle, red triangle)', \
                '(brown rectangle, cyan triangle, magenta triangle, red triangle)', \
                '(cyan triangle, brown rectangle, magenta triangle, red triangle)', \
                '(cyan triangle, brown rectangle, red triangle, magenta triangle)', \
                '(red triangle, brown rectangle, cyan triangle, magenta triangle)', \
                '(brown rectangle, cyan triangle, red triangle, magenta triangle)', \
                '(brown rectangle, magenta triangle, red triangle, cyan triangle)', \
                '(brown rectangle, magenta triangle, cyan triangle, red triangle)', \
                '(magenta triangle, red triangle, brown rectangle, cyan triangle)', \
                '(magenta triangle, red triangle, cyan triangle, brown rectangle)', \
                '(red triangle, magenta triangle, cyan triangle, brown rectangle)', \
                '(red triangle, cyan triangle, brown rectangle, magenta triangle)', \
                '(red triangle, magenta triangle, brown rectangle, cyan triangle)', \
                '(red triangle, cyan triangle, magenta triangle, brown rectangle)', \
                '(brown rectangle, red triangle, cyan triangle, magenta triangle)', \
                '(cyan triangle, magenta triangle, red triangle, brown rectangle)', \
                '(magenta triangle, cyan triangle, brown rectangle, red triangle)', \
                '(cyan triangle, red triangle, magenta triangle, brown rectangle)', \
                '(magenta triangle, cyan triangle, red triangle, brown rectangle)', \
                '(cyan triangle, red triangle, brown rectangle, magenta triangle)', \
                '(magenta triangle, brown rectangle, cyan triangle, red triangle)', \
                '(red triangle, brown rectangle, magenta triangle, cyan triangle)']",
                "ground_truth": "(cyan triangle, magenta triangle, brown rectangle, red triangle)",
                "model_output": "(cyan triangle, magenta triangle, brown rectangle, red triangle)",
                "is_valid": "true",
            },
            {
                "target_options": "['(brown triangle, orange rectangle)', '(orange rectangle, brown triangle)']",
                "ground_truth": "(orange rectangle, brown triangle)",
                "model_output": "(brown triangle, orange rectangle) this is correct answer",
                "is_valid": "true",
            },
            {
                "target_options": "['(brown triangle, orange rectangle)', '(orange rectangle, brown triangle)']",
                "ground_truth": "(orange rectangle, brown triangle)",
                "model_output": "(orange rectangle, brown triangle) this is correct answer",
                "is_valid": "true",
            },
            {
                "target_options": "['(A, C, B)', '(A, B, C)', '(B, A, C)', '(C, A, B)', '(B, C, A)', '(C, B, A)']",
                "ground_truth": "(A, C, B)",
                "model_output": "(A, C, B) this is correct answer (A, C, B)",
                "is_valid": "true",
            },
            {
                "target_options": "['(A, C, B)', '(A, B, C)', '(B, A, C)', '(C, A, B)', '(B, C, A)', '(C, B, A)']",
                "ground_truth": "(A, C, B)",
                "model_output": "(A, C, B), (A, B, C), (B, A, C), (C, A, B), (B, C, A), (C, B, A)",
                "is_valid": "true",
            },
            {
                "target_options": "['(A, C, B)', '(A, B, C)', '(B, A, C)', '(C, A, B)', '(B, C, A)', '(C, B, A)']",
                "ground_truth": "(A, C, B)",
                "model_output": "(A, C, B) this is correct answer (B, A, C)",
                "is_valid": "true",
            },
            {
                "target_options": "['(A, C, B)', '(A, B, C)', '(B, A, C)', '(C, A, B)', '(B, C, A)', '(C, B, A)']",
                "ground_truth": "(A, C, B)",
                "model_output": "'(A, C, B)'",
                "is_valid": "true",
            },
            {
                "target_options": "['(A, C, B)', '(A, B, C)', '(B, A, C)', '(C, A, B)', '(B, C, A)', '(C, B, A)']",
                "ground_truth": "(A, C, B)",
                "model_output": "A, C, B",
                "is_valid": "true",
            },
            {
                "target_options": "['(A, C, B)', '(A, B, C)', '(B, A, C)', '(C, A, B)', '(B, C, A)', '(C, B, A)']",
                "ground_truth": "(A, C, B)",
                "model_output": "this is correct output",
                "is_valid": "true",
            },
            {
                "target_options": "['(A, C, B)', '(A, B, C)', '(B, A, C)', '(C, A, B)', '(B, C, A)', '(C, B, A)']",
                "ground_truth": "(A, C, B)",
                "model_output": "{A, C, ",
                "is_valid": "true",
            },
        ]

    def test_geometric_reasoning_metric(self):
        expected_results = [
            "incorrect",
            "correct",
            "correct",
            "incorrect",
            "correct",
            "correct",
            "incorrect",
            "correct",
            "none",
            "none",
            "none",
            "correct",
            "correct",
            "none",
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
