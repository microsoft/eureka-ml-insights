# write unit tests for the classes in data_utils/ba_calendar_utils.py

import logging
import math
import unittest

import numpy as np
import pandas as pd

from eureka_ml_insights.data_utils.ba_calendar_utils import BA_Calendar_ExtractAnswer

log = logging.getLogger("BACalendar_ExtractAnswer_tests")

class TestBACalExtractAnswer(unittest.TestCase):
    def setUp(self):
        testcases = [
            "thinking steps ## Final Answer:\n 25",
            "thinking steps ## Final Answer: a, Final Answer: b, Final Answer: x",
            "thinking steps ## Final Answer: a, Final Answer: ball, Final Answer: xin",
        ]
        self.df = pd.DataFrame(
            {
                "raw_output": testcases, 
                "model_output": ["test"]*len(testcases),
            }
        )

    def test_labelparsing(self):
        transform = BA_Calendar_ExtractAnswer("raw_output", "model_output")
        transform.transform(self.df)
        # Check values, accounting for NaN
        expected_values = [
            "25",
            "x",
            "xin",
        ]
        np.testing.assert_array_equal(self.df["model_output"].values, expected_values)


if __name__ == "__main__":
    unittest.main()