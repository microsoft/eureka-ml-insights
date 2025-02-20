# write unit tests for the classes in data_utils/transform.py

import logging
import unittest

import numpy as np
import pandas as pd

from eureka_ml_insights.data_utils.gsm8k_utils import GSM8KExtractAnswer

log = logging.getLogger("GSM8K_ExtractAnswer_tests")

class TestGSM8KAnswerExtract(unittest.TestCase):
    def setUp(self):
        self.testcases = {
            "#### 1": 1.0,
            "test #### 3": 3.0,
            "#### 0": 0.0,
            "6%": float("nan"),
            "%6": float("nan"),
            "t": float("nan"),
            "10": float("nan"),
            "": float("nan")
        }
        self.df = pd.DataFrame(columns=["raw_output", "model_output"])
        self.df["raw_output"] = self.testcases.keys()


    def test_answerextraction(self):
        transform = GSM8KExtractAnswer("raw_output", "model_output")
        transform.transform(self.df)
        # Check values, accounting for NaN
        np.testing.assert_array_equal(self.df["model_output"].values, list(self.testcases.values()))


if __name__ == "__main__":
    unittest.main()