# write unit tests for the classes in data_utils
import os
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from eureka_ml_insights.data_utils import (
    ColumnMatchMapTransform,
    ColumnRename,
    HFDataReader,
    ImputeNA,
    JinjaPromptTemplate,
    MajorityVoteTransform,
    MapStringsTransform,
    MMDataLoader,
    MultiplyTransform,
    RegexTransform,
    ReplaceStringsTransform,
    RunPythonTransform,
    SequenceTransform,
    ShuffleColumnsTransform,
    TokenCounterTransform,
)


class TestDataTransform(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"A": [1, 2, None], "B": ["a", "b", "c"], "D": ["a", "aa", "cc"]})

    def test_column_rename(self):
        transform = ColumnRename({"A": "X"})
        result = transform.transform(self.df)
        self.assertListEqual(list(result.columns), ["X", "B", "D"])

    def test_impute_na(self):
        transform = ImputeNA("A", 0)
        result = transform.transform(self.df)
        self.assertListEqual(list(result["A"]), [1, 2, 0])

    def test_replace_strings_transform(self):
        transform = ReplaceStringsTransform(["B"], {"a": "A", "b": "B"}, case=False)
        result = transform.transform(self.df)
        self.assertListEqual(list(result["B"]), ["A", "B", "c"])

    def test_map_strings_transform(self):
        transform = MapStringsTransform(["D"], {"a": "X", "aa": "Y"})
        result = transform.transform(self.df)
        self.assertListEqual(list(result["D"]), ["X", "Y", np.nan])

    def test_sequence_transform(self):
        transform1 = ColumnRename({"A": "X"})
        transform2 = ImputeNA("X", 0)
        sequence_transform = SequenceTransform([transform1, transform2])
        result = sequence_transform.transform(self.df)
        self.assertListEqual(list(result.columns), ["X", "B", "D"])
        self.assertListEqual(list(result["X"]), [1, 2, 0])

    def test_run_python_transform(self):
        python_code = "df['C'] = df['A'].fillna(0)"
        transform = RunPythonTransform(python_code)
        result = transform.transform(self.df)
        self.assertListEqual(list(result.columns), ["A", "B", "D", "C"])
        self.assertListEqual(list(result["C"]), [1, 2, 0])

    def test_regex_transform(self):
        model_output = ["B"]
        prompt_pattern = r"\b\d+(?:\.\d+)?\b"
        model_data_loc1 = "The correctness score is 1.0 (totally right)"
        model_data_loc2 = "The prediction is partially correct, so the correctness score for the prediction is 0.8."
        model_data_loc3 = "The prediction is not available so can not give a correctness score."
        df1 = pd.DataFrame({"A": [1, 2, None], "B": [model_data_loc1, model_data_loc2, model_data_loc3]})
        transform = SequenceTransform(
            [
                RegexTransform(model_output, prompt_pattern, ignore_case=True),
                ImputeNA("B", "0.0"),
            ]
        )
        result = transform.transform(df1)
        self.assertListEqual(list(result["B"]), ["1.0", "0.8", "0.0"])

    def test_multiply_transform(self):
        n_repeats = 2
        transform = MultiplyTransform(n_repeats=n_repeats)
        result = transform.transform(self.df)
        self.assertListEqual(list(result["B"]), ["a", "b", "c", "a", "b", "c"])
        self.assertEqual(len(result["A"]), len(self.df) * n_repeats)

    def test_majorityvote_transform(self):
        df1 = pd.DataFrame(
            {"data_point_id": [1, 1, 1, 2, 2, 2, 3, 3, 3], "model_output": [100, 100, 99, 5, 4, 1, 2, np.nan, np.nan]}
        )
        transform = MajorityVoteTransform()
        result = transform.transform(df1)
        self.assertListEqual(
            list(result),
            list(
                pd.DataFrame(
                    {
                        "data_point_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                        "model_output": [100, 100, 99, 1, 5, 5, 2, np.nan, np.nan],
                        "majority_vote": [100, 100, 100, 5, 5, 5, 2, 2, 2],
                    }
                )
            ),
        )


class TestTokenCounterTransform(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"A": ["tiktoken is great!"], "B": ["antidisestablishmentarianism"]})

    def test_token_transform(self):
        transform = TokenCounterTransform("A")
        result = transform.transform(self.df)
        # make sure the token count is correct
        self.assertEqual(result["A_token_count"][0], 6)
        # make sure the other columns are not affected
        self.assertEqual(result["B"][0], self.df.loc[0, "B"])
        self.assertEqual(result["A"][0], self.df.loc[0, "A"])

    def test_token_transform_multi_columns(self):
        transform = TokenCounterTransform(["A", "B"])
        result = transform.transform(self.df)
        # make sure the token count is correct
        self.assertEqual(result["A_token_count"][0], 6)
        self.assertEqual(result["B_token_count"][0], 6)
        # make sure the other columns are not affected
        self.assertEqual(result["B"][0], self.df.loc[0, "B"])
        self.assertEqual(result["A"][0], self.df.loc[0, "A"])


class JinjaPromptTemplateTest(unittest.TestCase):
    def setUp(self):
        self.template_path = "tests/data_utils_tests/test_prompt_template.jinja"
        self.values = {"country": "Iran", "city": "Tehran"}

    def test_jinja_prompt_template(self):
        template = JinjaPromptTemplate(self.template_path)
        result = template.create(self.values)
        self.assertEqual(result, 'The prompt is: "What is the capital of Iran?" The correct answer is Tehran.')


class TestHFDataReader(unittest.TestCase):
    """Testing HFDataReader for dataset loading and local image caching with MMMU."""

    def setUp(self) -> None:
        self.data_reader = HFDataReader("MMMU/MMMU", "dev", ["Art"])

    def test_hf_data_readers(self):
        # load dataset
        df = self.data_reader.load_dataset()
        # make sure expected colunmns are present
        self.assertIn("image_1", df.columns.tolist())
        # check that a cached image exists, to test if local caching worked
        self.assertTrue(os.path.exists(df["image_1"].values[0]))


class TestMMDataLoader(unittest.TestCase):
    """Testing MMDataLoader with single image, local sample dataset."""

    def setUp(self) -> None:
        self.data_loader = MMDataLoader("./tests/test_assets/mm_test_data.jsonl", "./tests/test_assets/")

    def test_mm_loader(self):
        with self.data_loader:
            for _, model_inputs, model_args, model_kwargs in self.data_loader:
                self.assertTrue(isinstance(model_inputs["query_images"][0], Image.Image))


class TestShuffleColumns(unittest.TestCase):
    """Testing the ShuffleColumnsTransform used in MCQ benchmarks to shuffle answer choices."""

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": ["a", "b", "c", "d", "e"],
                "C": [-10, -20, -30, -40, -50],
                "D": ["hi", "how", "are", "you", "?"],
            }
        )
        self.shuffle_transform = ShuffleColumnsTransform(columns=["A", "B", "C"])

    def test_shuffle_columns_values(self):
        # Apply the transformation twice with two different generators that have different seeds
        rng_1 = np.random.default_rng(42)
        self.shuffle_transform.rng = rng_1
        transformed_df_1 = self.shuffle_transform.transform(self.df.copy())
        rng_2 = np.random.default_rng(0)
        self.shuffle_transform.rng = rng_2
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


class TestColMatchMap(unittest.TestCase):
    """
    Testing the ColumnMatchMapTransform used in MCQ benchmarks to store the letter of the correct
    answer choice.
    """

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
            self.col_match_map_transform = ColumnMatchMapTransform(
                key_col=val["key_col"], new_col="ground_truth", columns=val["cols"]
            )
            df = val["df"]
            df = self.col_match_map_transform.transform(df)
            self.assertEqual(list(df["ground_truth"]), val["ground_truth"])


if __name__ == "__main__":
    unittest.main()
