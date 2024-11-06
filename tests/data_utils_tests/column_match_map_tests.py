import unittest

import numpy as np
import pandas as pd

from eureka_ml_insights.data_utils.transform import ColumnMatchMap


class TestColMatchMap(unittest.TestCase):
    def setUp(self):
        # Seed the random number generator for reproducibility
        np.random.seed(42)

        # Sample DataFrame
        self.values = [
            {
                "df": pd.DataFrame(
                    {
                        "A": [1, 2, 3, 4, "e"],
                        "B": ["a", "b", "c", "d", "e"],
                        "C": ["a", -20, -30, "d", -50],
                        "D": ["hi", "b", "c", "you", "?"],
                    }
                ),
                "cols": ["A", "C", "D"],
                "key_col": "B",
                "ground_truth": ["C", "D", "D", "C", "A"],
            }
        ]

    def test_col_match_map(self):
        for val in self.values:
            self.col_match_map_transform = ColumnMatchMap(
                key_col=val["key_col"], new_col="ground_truth", columns=val["cols"]
            )
            df = val["df"]
            df = self.col_match_map_transform.transform(df)
            self.assertEqual(list(df["ground_truth"]), val["ground_truth"])


if __name__ == "__main__":
    unittest.main()
