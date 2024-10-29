import os
import time
import unittest

import pandas as pd

from eureka_ml_insights.configs import (
    DataJoinConfig,
    DataProcessingConfig,
    DataSetConfig,
    InferenceConfig,
    ModelConfig,
    PromptProcessingConfig,
    create_logdir,
)
from eureka_ml_insights.core import (
    DataJoin,
    DataProcessing,
    Inference,
    PromptProcessing,
)
from eureka_ml_insights.data_utils import (
    ColumnRename,
    DataReader,
    ReplaceStringsTransform,
    RunPythonTransform,
    SequenceTransform,
)
from tests.test_utils import TestDataLoader, TestModel


class TestPromptProcessing(unittest.TestCase):
    def setUp(self) -> None:
        self.log_dir = create_logdir("TestPromptProcessing")
        self.config = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": "./sample_data/sample_data.jsonl",
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={"target_text": "ground_truth"}),
                            ReplaceStringsTransform(columns=["query_text"], mapping={"\n": "*N*"}, case=True),
                        ]
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../eureka_ml_insights/prompt_templates/basic.jinja"
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
            output_data_columns=["query_text", "ground_truth"],
            ignore_failure=False,
        )
        component = PromptProcessing.from_config(self.config)
        component.run()

    def test_prompt_processing(self):
        self.assertTrue(os.path.exists(self.config.output_dir))
        self.assertTrue(os.path.exists(os.path.join(self.config.output_dir, "transformed_data.jsonl")))
        df = pd.read_json(os.path.join(self.config.output_dir, "transformed_data.jsonl"), lines=True)
        # prompt, prompt_hash, and uid columns are added by the prompt processing component
        self.assertSetEqual(
            set(df.columns.tolist()), set(["query_text", "ground_truth", "prompt_hash", "prompt", "uid"])
        )
        self.assertEqual(df["query_text"].str.contains("\n").sum(), 0)


class TestDataProcessing(unittest.TestCase):
    def setUp(self) -> None:
        self.log_dir = create_logdir("TestDataProcessing")
        self.config = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": "./sample_data/sample_data.jsonl",
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={"target_text": "ground_truth"}),
                            ReplaceStringsTransform(columns=["query_text"], mapping={"\n": "*N*"}, case=True),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
            output_data_columns=["query_text", "ground_truth"],
        )
        component = DataProcessing.from_config(self.config)
        component.run()

    def test_data_processing(self):
        self.assertTrue(os.path.exists(self.config.output_dir))
        self.assertTrue(os.path.exists(os.path.join(self.config.output_dir, "transformed_data.jsonl")))
        df = pd.read_json(os.path.join(self.config.output_dir, "transformed_data.jsonl"), lines=True)
        self.assertSetEqual(set(df.columns.tolist()), set(["query_text", "ground_truth"]))
        self.assertEqual(df["query_text"].str.contains("\n").sum(), 0)


class TestDataJoin(unittest.TestCase):
    def setUp(self) -> None:
        self.log_dir = create_logdir("TestDataJoin")
        self.config = DataJoinConfig(
            component_type=DataJoin,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": "./sample_data/sample_data.jsonl",
                    "format": ".jsonl",
                    "transform": RunPythonTransform("df['images'] = df['images'].apply(lambda x: x[0])"),
                },
            ),
            other_data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": "./sample_data/sample_data2.jsonl",
                    "format": ".jsonl",
                    "transform": 
                    SequenceTransform([RunPythonTransform("df['images'] = df['images'].apply(lambda x: x[0])"),
                                       ColumnRename(name_mapping={"prompt": "prompt_2"})]),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_join_output"),
            output_data_columns=["images", "query_text", "ground_truth"],
            pandas_merge_args={"on": "images", "how": "inner"},
        )
        component = DataJoin.from_config(self.config)
        component.run()

    def test_data_join(self):
        self.assertTrue(os.path.exists(self.config.output_dir))
        self.assertTrue(os.path.exists(os.path.join(self.config.output_dir, "transformed_data.jsonl")))
        df = pd.read_json(os.path.join(self.config.output_dir, "transformed_data.jsonl"), lines=True)
        self.assertSetEqual(set(df.columns.tolist()), set(["images", "query_text", "ground_truth"]))
        # assert that no rows are lost
        n_lines = 0
        with open("./sample_data/sample_data.jsonl") as f:
            for _ in f:
                n_lines += 1
        self.assertEqual(len(df), n_lines)


class TestInference(unittest.TestCase):
    def setUp(self) -> None:
        self.log_dir = create_logdir("TestInference")
        self.config = InferenceConfig(
            component_type=Inference,
            data_loader_config=DataSetConfig(
                TestDataLoader,
                {
                    "path": "./tests/test_assets/transformed_data.jsonl",
                    "n_iter": 40,
                },
            ),
            model_config=ModelConfig(TestModel, {}),
            output_dir=os.path.join(self.log_dir, "model_output"),
            resume_from="./tests/test_assets/resume_from.jsonl",
        )
        component = Inference.from_config(self.config)
        component.run()

    def test_inference(self):
        # load inference results
        df = pd.read_json(os.path.join(self.config.output_dir, "inference_result.jsonl"), lines=True)
        df["is_valid"] = df["is_valid"].astype(bool)
        # load resume_from file
        resume_from_df = pd.read_json(self.config.resume_from, lines=True)
        resume_from_df["is_valid"] = resume_from_df["is_valid"].astype(bool)
        # join on uid
        merged_df = df.merge(resume_from_df, on="uid", suffixes=("_new", "_old"), how="left")
        merged_df["is_valid_old"] = merged_df[merged_df["is_valid_old"].isna()]["is_valid_old"] = False
        merged_df["is_valid_new"] = merged_df[merged_df["is_valid_new"].isna()]["is_valid_new"] = False
        # assert that when is_valid is true in resume_from, it's still true in the inference results
        merged_df["validity_match"] = merged_df.apply(lambda x: x["is_valid_old"] and x["is_valid_new"], axis=1)
        self.assertTrue(merged_df[merged_df["is_valid_old"]]["validity_match"].all())

        # assert that model_output is the same in resume_from and inference results when is_valid is true
        merged_df["model_output_match"] = merged_df["model_output_new"] == merged_df["model_output_old"]
        self.assertTrue(merged_df[merged_df["is_valid_old"]]["model_output_match"].all())

        # assert that number of rows in inference results is greater than or equal to resume_from
        self.assertGreaterEqual(len(df), len(resume_from_df))


class TestParallelInference(unittest.TestCase):
    def setUp(self) -> None:
        self.log_dir = create_logdir("TestInference")
        self.config = InferenceConfig(
            component_type=Inference,
            data_loader_config=DataSetConfig(
                TestDataLoader,
                {
                    "path": "./tests/test_assets/transformed_data.jsonl",
                    "n_iter": 40,
                },
            ),
            model_config=ModelConfig(TestModel, {}),
            output_dir=os.path.join(self.log_dir, "model_output"),
            resume_from="./tests/test_assets/resume_from.jsonl",
            max_concurrent=5,
        )
        component = Inference.from_config(self.config)
        component.run()

    def test_inference(self):
        # load inference results
        df = pd.read_json(os.path.join(self.config.output_dir, "inference_result.jsonl"), lines=True)
        df["is_valid"] = df["is_valid"].astype(bool)
        # load resume_from file
        resume_from_df = pd.read_json(self.config.resume_from, lines=True)
        resume_from_df["is_valid"] = resume_from_df["is_valid"].astype(bool)
        # join on uid
        merged_df = df.merge(resume_from_df, on="uid", suffixes=("_new", "_old"), how="left")
        merged_df["is_valid_old"] = merged_df[merged_df["is_valid_old"].isna()]["is_valid_old"] = False
        merged_df["is_valid_new"] = merged_df[merged_df["is_valid_new"].isna()]["is_valid_new"] = False
        # assert that when is_valid is true in resume_from, it's still true in the inference results
        merged_df["validity_match"] = merged_df.apply(lambda x: x["is_valid_old"] and x["is_valid_new"], axis=1)
        self.assertTrue(merged_df[merged_df["is_valid_old"]]["validity_match"].all())

        # assert that model_output is the same in resume_from and inference results when is_valid is true
        merged_df["model_output_match"] = merged_df["model_output_new"] == merged_df["model_output_old"]
        self.assertTrue(merged_df[merged_df["is_valid_old"]]["model_output_match"].all())

        # assert that number of rows in inference results is greater than or equal to resume_from
        self.assertGreaterEqual(len(df), len(resume_from_df))


class TestRateLimitedInference(unittest.TestCase):
    def setUp(self) -> None:
        self.log_dir = create_logdir("TestInference")
        self.config = InferenceConfig(
            component_type=Inference,
            data_loader_config=DataSetConfig(
                TestDataLoader,
                {
                    "path": "./tests/test_assets/transformed_data.jsonl",
                    "n_iter": 40,
                },
            ),
            model_config=ModelConfig(TestModel, {}),
            output_dir=os.path.join(self.log_dir, "model_output"),
            n_calls_per_min=20,
        )
        component = Inference.from_config(self.config)
        start = time.time()
        component.run()
        end = time.time()
        self.duration = end - start

    # skip this test if environment variable skip_slow_tests is set
    @unittest.skipIf(os.environ.get("skip_slow_tests"), "Skipping slow test")
    def test_inference(self):
        # load inference results
        df = pd.read_json(os.path.join(self.config.output_dir, "inference_result.jsonl"), lines=True)
        # assert that all inferences are valid
        self.assertTrue(df["is_valid"].all())
        # assert that there are 40 inference results
        self.assertEqual(len(df), 40)
        # assert that the duration is greater than 40/20 - 1 minutes
        self.assertGreaterEqual(self.duration, (40 / 20 - 1) * 60)


if __name__ == "__main__":
    unittest.main()
