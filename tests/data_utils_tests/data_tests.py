# write unit tests for the classes in data_utils
import os
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from eureka_ml_insights.data_utils import (
    ColumnRename,
    HFDataReader,
    ImputeNA,
    JinjaPromptTemplate,
    MapStringsTransform,
    MMDataLoader,
    MultiplyTransform,
    RegexTransform,
    ReplaceStringsTransform,
    RunPythonTransform,
    SequenceTransform,
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
                RegexTransform(model_output, prompt_pattern, case=True),
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
            for _, model_inputs in self.data_loader:
                self.assertTrue(isinstance(model_inputs[1][0], Image.Image))


if __name__ == "__main__":
    unittest.main()
