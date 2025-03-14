# write unit tests for the classes in data_utils/omni_math_utils.py

import logging
import math
import unittest

import numpy as np
import pandas as pd

from eureka_ml_insights.data_utils.omni_math_utils import Omni_Math_ParseLabel, Omni_Math_ParseSolution

log = logging.getLogger("OmniMath_ParseLabel_tests")

class TestOmniMathParseLabel(unittest.TestCase):
    def setUp(self):
        testcases = [
            "thinking steps ## Student Final Answer\n 25\n ## Equivalence Judgement\n true\n ## Justification\n Correct answer",
            "thinking steps ## Student Final Answer\n None\n ## Equivalence Judgement\n false\n ## Justification\n Incorrect answer",
            "thinking steps ## student final Answer\n twenty\n ## Equivalence Judgement\n \n ## Justification\n No answer",
            "thinking steps ## Student Final Answer\n 42\n ## Equivalence Judgement\n true\n ## Justification\n Correct answer",
            "thinking steps ## Student Final Answer\n 0\n ## Equivalence Judgement\n false\n ## Justification\n Incorrect answer",
            "thinking steps ## Student Final Answer\n 3.14\n ## Equivalence Judgement\n true\n ## Justification\n Correct answer",
            "thinking steps ## Student Final Answer\n -1\n ## Equivalence Judgement\n false\n ## Justification\n Incorrect answer",
            "thinking steps ## Student Final Answer\n 100\n ## Equivalence Judgement\n true\n ## Justification\n Correct answer",
        ]
        self.df = pd.DataFrame(
            {
                "raw_output": testcases,
                "OmniMath_Correctness": ["0"]*len(testcases), 
                "model_output": ["test"]*len(testcases),
            }
        )

    def test_labelparsing(self):
        transform = Omni_Math_ParseLabel("raw_output", "OmniMath_Correctness")
        transform.transform(self.df)
        # Check values, accounting for NaN
        expected_values = [1, 0, math.nan, 1, 0, 1, 0, 1]
        np.testing.assert_array_equal(self.df["OmniMath_Correctness"].values, expected_values)

    def test_solutionparsing(self):
        transform = Omni_Math_ParseSolution("raw_output", "model_output")
        transform.transform(self.df)
        # Check values
        expected_values = ["25", "None", "twenty", "42", "0", "3.14", "-1", "100"]
        np.testing.assert_array_equal(self.df["model_output"].values, expected_values)


if __name__ == "__main__":
    unittest.main()
