# write unit tests for the classes in data_utils/transform.py

import logging
import unittest

import numpy as np
import pandas as pd

from eureka_ml_insights.data_utils.aime_utils import AIMEExtractAnswer

log = logging.getLogger("AIME_ExtractAnswer_tests")

class TestAIMEAnswerExtract(unittest.TestCase):
    def setUp(self):
        testcases = ["1", "3", "0$", "6%", "5", "-1", "t", "10", ""]
        self.df = pd.DataFrame(
            {
                "C": ["Final Answer: {0} ".format(s) for s in testcases],
                "D": ["0", "0", "0", "0", "0", "0", "0", "0", "0"],
            }
        )

    def test_answerextraction(self):
        transform = AIMEExtractAnswer("C", "D")
        transform.transform(self.df)
        # Check values, accounting for NaN
        expected_values = [1.0, 3.0, 0.0, 0.06, 5.0, -1.0, float("nan"), 10.0, float("nan")]
        np.testing.assert_array_equal(self.df["D"].values, expected_values)


if __name__ == "__main__":
    unittest.main()
