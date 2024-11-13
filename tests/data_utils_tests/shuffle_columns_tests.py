import unittest

import numpy as np
import pandas as pd

from eureka_ml_insights.data_utils.transform import ShuffleColumns


class TestShuffleColumns(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
                    {
                        "A": [1, 2, 3, 4, 5],
                        "B": ["a", "b", "c", "d", "e"],
                        "C": [-10, -20, -30, -40, -50],
                        "D": ["hi", "how", "are", "you", "?"],
                    }
                )
        self.shuffle_transform = ShuffleColumns(columns=["A", "B", "C"])

    def test_shuffle_columns_values(self):
        # Apply the transformation twice
        np.random.seed(42)
        transformed_df_1 = self.shuffle_transform.transform(self.df.copy())
        np.random.seed(0)
        transformed_df_2 = self.shuffle_transform.transform(self.df.copy())

        # Columns that should remain unchanged
        unshuffled_columns = [col for col in self.df.columns if col not in self.shuffle_transform.columns]

        # Ensure each row has the same set of values in the shuffled columns after both transformations
        for _, row in self.df.iterrows():
            original_values = set(row[self.shuffle_transform.columns])

            # Get the transformed row values for both shuffles
            transformed_values_1 = set(transformed_df_1.loc[row.name, self.shuffle_transform.columns])
            transformed_values_2 = set(transformed_df_2.loc[row.name, self.shuffle_transform.columns])

            # Check that each transformed row has the same set of values as the original
            self.assertEqual(original_values, transformed_values_1)
            self.assertEqual(original_values, transformed_values_2)

            # Verify that the order is different between the two shuffles
            self.assertNotEqual(
                tuple(transformed_df_1.loc[row.name, self.shuffle_transform.columns]),
                tuple(transformed_df_2.loc[row.name, self.shuffle_transform.columns]),
            )

        # Ensure unshuffled columns remain the same in both transformations
        for col in unshuffled_columns:
            pd.testing.assert_series_equal(self.df[col], transformed_df_1[col], check_exact=True)
            pd.testing.assert_series_equal(self.df[col], transformed_df_2[col], check_exact=True)

if __name__ == "__main__":
    unittest.main()
