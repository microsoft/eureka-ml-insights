import unittest

from parameterized import parameterized

from eureka_ml_insights.metrics.live_code_bench import pass_at_k_estimator


class EstimatePassAtKTest(unittest.TestCase):
    """Tests for estimate_pass_at_k function."""

    @parameterized.expand([
        # num_attempts, num_correct, k, expected_pass_at_k

        # No correct attempts
        (10, 0, 1, 0.0),

        # Half correct attempts with pass@1
        (10, 5, 1, 0.5),

        # Half correct attempts with pass@5
        (10, 5, 5, 0.996032),

        # All correct attempts with pass@1
        (10, 10, 1, 1.0),

        # All correct attempts with pass@5
        (10, 10, 5, 1.0),

        # K greater than num_attempts with some correct attempts
        (5, 3, 10, 1.0),

        # K greater than num_attempts with no correct attempts
        (5, 0, 10, 0.0),
    ])
    def test_estimate_pass_at_k(
        self,
        num_attempts: int,
        num_correct: int,
        k: int,
        expected_pass_at_k: float,
    ):
        """Tests the estimate_pass_at_k function with various inputs."""
        result = pass_at_k_estimator.estimate_pass_at_k(
            num_attempts=num_attempts,
            num_correct=num_correct,
            k=k,
        )
        self.assertAlmostEqual(result, expected_pass_at_k, places=6)
    
    def test_num_correct_exceeds_attempts(self):
        """Tests that ValueError is raised when num_correct > num_attempts."""
        with self.assertRaisesRegex(
            ValueError,
            "Number of correct attempts cannot exceed total attempts."):
            pass_at_k_estimator.estimate_pass_at_k(
                num_attempts=5, num_correct=6, k=1)

    @parameterized.expand([
        (0,),
        (-1,),
    ])
    def test_invalid_k_value(self, k: int):
        """Tests that ValueError is raised for invalid k values."""
        with self.assertRaisesRegex(
            ValueError,
            "K must be a positive integer."):
            pass_at_k_estimator.estimate_pass_at_k(
                num_attempts=5, num_correct=3, k=k)


if __name__ == "__main__":
    unittest.main()
