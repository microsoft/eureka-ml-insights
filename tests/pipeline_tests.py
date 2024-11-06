import logging
import os
import sys
import unittest
from pathlib import Path

import jsonlines

# setup loggers
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())  # noqa
sys.path.insert(0, path)  # noqa

from eureka_ml_insights.configs import (
    AIME_PIPELINE,
    DNA_PIPELINE,
    GEOMETER_PIPELINE,
    KITAB_ONE_BOOK_CONSTRAINT_PIPELINE,
    MAZE_PIPELINE,
    MAZE_TEXTONLY_PIPELINE,
    MMMU_BASELINE_PIPELINE,
    OBJECT_DETECTION_SINGLE_PIPELINE,
    OBJECT_RECOGNITION_SINGLE_PIPELINE,
    SPATIAL_GRID_PIPELINE,
    SPATIAL_GRID_TEXTONLY_PIPELINE,
    SPATIAL_MAP_PIPELINE,
    SPATIAL_MAP_TEXTONLY_PIPELINE,
    SPATIAL_REASONING_SINGLE_PIPELINE,
    VISUAL_PROMPTING_SINGLE_PIPELINE,
    Drop_Experiment_Pipeline,
    IFEval_PIPELINE,
    MetricConfig,
    ModelConfig,
    ToxiGen_Discriminative_PIPELINE,
)
from eureka_ml_insights.core import Pipeline
from eureka_ml_insights.data_utils.transform import (
    RunPythonTransform,
    SamplerTransform,
    SequenceTransform,
)
from tests.test_utils import (
    DetectionTestModel,
    DNAEvaluationInferenceTestModel,
    GenericTestModel,
    GeometricReasoningTestModel,
    KitabTestModel,
    MultipleChoiceTestModel,
    SpatialReasoningTestModel,
    TestDataLoader,
    TestHFDataReader,
    TestKitabMetric,
    TestMMDataLoader,
    ToxiGenTestModel,
)

N_ITER = 2


class TEST_SPATIAL_REASONING_PIPELINE(SPATIAL_REASONING_SINGLE_PIPELINE):
    # Test config the spatial reasoning benchmark with the SpatialReasoningTestModel and TestMMDataLoader
    # with small sample data and a test model
    def configure_pipeline(self):
        model_config = ModelConfig(SpatialReasoningTestModel, {})
        config = super().configure_pipeline(model_config=model_config)
        self.data_processing_comp.data_reader_config.class_name = TestHFDataReader
        self.inference_comp = config.component_configs[1]
        self.inference_comp.data_loader_config.class_name = TestMMDataLoader
        self.inference_comp.data_loader_config.init_args["n_iter"] = N_ITER
        return config


class TEST_OBJECT_DETECTION_PIPELINE(OBJECT_DETECTION_SINGLE_PIPELINE):
    # Test config the object detection benchmark with the DetectionTestModel and TestMMDataLoader
    # with small sample data and a test model
    def configure_pipeline(self, resume_from=None):
        model_config = ModelConfig(DetectionTestModel, {})
        config = super().configure_pipeline(model_config=model_config)
        self.data_processing_comp.data_reader_config.class_name = TestHFDataReader
        self.inference_comp.data_loader_config.class_name = TestMMDataLoader
        self.inference_comp.data_loader_config.init_args["n_iter"] = N_ITER
        return config


class TEST_VISUAL_PROMPTING_PIPELINE(VISUAL_PROMPTING_SINGLE_PIPELINE):
    # Test config the visual prompting benchmark with the GenericTestModel and TestMMDataLoader
    # with small sample data and a test model
    def configure_pipeline(self, resume_from=None):
        model_config = ModelConfig(GenericTestModel, {})
        config = super().configure_pipeline(model_config=model_config)
        self.data_processing_comp.data_reader_config.class_name = TestHFDataReader
        self.inference_comp.data_loader_config.class_name = TestMMDataLoader
        self.inference_comp.data_loader_config.init_args["n_iter"] = N_ITER
        return config


class TEST_OBJECT_RECOGNITION_PIPELINE(OBJECT_RECOGNITION_SINGLE_PIPELINE):
    # Test config the object recognition benchmark with the GenericTestModel and TestMMDataLoader
    # with small sample data and a test model
    def configure_pipeline(self, resume_from=None):
        model_config = ModelConfig(GenericTestModel, {})
        config = super().configure_pipeline(model_config=model_config)
        self.data_processing_comp.data_reader_config.class_name = TestHFDataReader
        self.inference_comp.data_loader_config.class_name = TestMMDataLoader
        self.inference_comp.data_loader_config.init_args["n_iter"] = N_ITER
        return config


class TEST_SPATIAL_GRID_PIPELINE(SPATIAL_GRID_PIPELINE):
    # Test config the spatial grid counting benchmark with the TestMMDataLoader
    # with small sample data and a test model
    def configure_pipeline(self, resume_from=None):
        model_config = ModelConfig(GenericTestModel, {})
        config = super().configure_pipeline(model_config=model_config)
        self.data_processing_comp.data_reader_config.class_name = TestHFDataReader
        self.inference_comp.data_loader_config.class_name = TestMMDataLoader
        self.inference_comp.data_loader_config.init_args["n_iter"] = N_ITER
        return config


class TEST_SPATIAL_GRID_TEXTONLY_PIPELINE(SPATIAL_GRID_TEXTONLY_PIPELINE):
    # Test config the spatial grid counting benchmark textonly version with the TestDataLoader
    # with small sample data and a test model
    def configure_pipeline(self, resume_from=None):
        model_config = ModelConfig(GenericTestModel, {})
        config = super().configure_pipeline(model_config=model_config)
        self.data_processing_comp.data_reader_config.class_name = TestHFDataReader
        self.inference_comp.data_loader_config.class_name = TestDataLoader
        self.inference_comp.data_loader_config.init_args["n_iter"] = N_ITER
        return config


class TEST_SPATIAL_MAP_PIPELINE(SPATIAL_MAP_PIPELINE):
    # Test config the spatial map benchmark with the TestMMDataLoader
    # with small sample data and a test model
    def configure_pipeline(self, resume_from=None):
        model_config = ModelConfig(GenericTestModel, {"model_name": "generic_test_model"})
        config = super().configure_pipeline(model_config=model_config)
        self.data_processing_comp.data_reader_config.class_name = TestHFDataReader
        self.inference_comp.data_loader_config.class_name = TestMMDataLoader
        self.inference_comp.data_loader_config.init_args["n_iter"] = N_ITER
        return config


class TEST_SPATIAL_MAP_TEXTONLY_PIPELINE(SPATIAL_MAP_TEXTONLY_PIPELINE):
    # Test config the spatial map benchmark textonly version with the TestDataLoader
    # with small sample data and a test model
    def configure_pipeline(self, resume_from=None):
        model_config = ModelConfig(GenericTestModel, {"model_name": "generic_test_model"})
        config = super().configure_pipeline(model_config=model_config)
        self.data_processing_comp.data_reader_config.class_name = TestHFDataReader
        self.inference_comp.data_loader_config.class_name = TestDataLoader
        self.inference_comp.data_loader_config.init_args["n_iter"] = N_ITER
        return config


class TEST_MAZE_PIPELINE(MAZE_PIPELINE):
    # Test config the maze benchmark with the TestMMDataLoader
    # with small sample data and a test model
    def configure_pipeline(self, resume_from=None):
        model_config = ModelConfig(GenericTestModel, {})
        config = super().configure_pipeline(model_config=model_config)
        self.data_processing_comp.data_reader_config.class_name = TestHFDataReader
        self.inference_comp.data_loader_config.class_name = TestMMDataLoader
        self.inference_comp.data_loader_config.init_args["n_iter"] = N_ITER
        return config


class TEST_MAZE_TEXTONLY_PIPELINE(MAZE_TEXTONLY_PIPELINE):
    # Test config the maze benchmark textonly version with the TestDataLoader
    # with small sample data and a test model
    def configure_pipeline(self, resume_from=None):
        model_config = ModelConfig(GenericTestModel, {})
        config = super().configure_pipeline(model_config=model_config)
        self.data_processing_comp.data_reader_config.class_name = TestHFDataReader
        self.inference_comp.data_loader_config.class_name = TestDataLoader
        self.inference_comp.data_loader_config.init_args["n_iter"] = N_ITER
        return config


class TEST_KITAB_ONE_BOOK_CONSTRAINT_PIPELINE(KITAB_ONE_BOOK_CONSTRAINT_PIPELINE):
    def configure_pipeline(self):
        config = super().configure_pipeline(model_config=ModelConfig(KitabTestModel, {}))
        inference_comp = config.component_configs[1]
        inference_comp.data_loader_config.class_name = TestDataLoader
        inference_comp.data_loader_config.init_args["n_iter"] = N_ITER
        self.evalreporting_comp.metric_config = MetricConfig(TestKitabMetric)
        self.evalreporting_comp.aggregator_configs[0].init_args["column_names"] = [
            "TestKitabMetric_satisfied_rate",
            "TestKitabMetric_unsatisfied_rate",
            "TestKitabMetric_not_from_author_rate",
            "TestKitabMetric_completeness",
            "TestKitabMetric_all_correct",
        ]
        self.evalreporting_comp.aggregator_configs[1].init_args["column_names"] = [
            "TestKitabMetric_satisfied_rate",
            "TestKitabMetric_unsatisfied_rate",
            "TestKitabMetric_not_from_author_rate",
            "TestKitabMetric_completeness",
            "TestKitabMetric_all_correct",
        ]
        return config


class TEST_GEOMETRIC_REASONING_PIPELINE(GEOMETER_PIPELINE):
    # Test config the spatial reasoning benchmark with the TestDataLoader
    # with small sample data and a test model
    def configure_pipeline(self):
        model_config = ModelConfig(GeometricReasoningTestModel, {})
        config = super().configure_pipeline(model_config=model_config)
        data_processing_comp = config.component_configs[0]
        data_processing_comp.data_reader_config.class_name = TestHFDataReader
        inference_comp = config.component_configs[1]
        inference_comp.data_loader_config.class_name = TestDataLoader
        inference_comp.data_loader_config.init_args["n_iter"] = N_ITER
        return config


class TEST_DNA_PIPELINE(DNA_PIPELINE):
    # Test config the Do Not Answer benchmark with TestModel and TestDataLoader
    def configure_pipeline(self):
        model_config_eval = ModelConfig(DNAEvaluationInferenceTestModel, {})
        config = super().configure_pipeline(model_config=ModelConfig(GenericTestModel, {}))
        self.inference_comp.data_loader_config.class_name = TestDataLoader
        self.inference_comp.data_loader_config.init_args = {
            "path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl"),
            "n_iter": N_ITER,
        }
        self.eval_inference_comp.model_config = model_config_eval
        self.eval_inference_comp.data_loader_config.class_name = TestDataLoader
        self.eval_inference_comp.data_loader_config.init_args = {
            "path": os.path.join(self.eval_data_pre_processing_comp.output_dir, "transformed_data.jsonl"),
            "n_iter": N_ITER,
        }
        return config


class TEST_IFEval_PIPELINE(IFEval_PIPELINE):
    # Test config the IFEval benchmark with TestModel and TestDataLoader
    def configure_pipeline(self):
        config = super().configure_pipeline(model_config=ModelConfig(GenericTestModel, {}))
        self.data_processing_comp.data_reader_config.init_args["transform"] = SequenceTransform(
            [
                RunPythonTransform("df['instruction_id_list_copy'] = df.loc[:, 'instruction_id_list']"),
                RunPythonTransform("df = df.explode(['instruction_id_list_copy'])"),
                SamplerTransform(sample_count=N_ITER, random_seed=99, stratify_by="instruction_id_list_copy"),
            ]
        )
        return config


class TEST_TOXIGEN_PIPELINE(ToxiGen_Discriminative_PIPELINE):
    def configure_pipeline(self):
        config = super().configure_pipeline(model_config=ModelConfig(ToxiGenTestModel, {}))
        self.inference_comp.data_loader_config.class_name = TestDataLoader
        self.inference_comp.data_loader_config.init_args = {
            "path": os.path.join(self.data_pre_processing.output_dir, "transformed_data.jsonl"),
            "n_iter": N_ITER,
        }
        return config


class TEST_MMMU_PIPELINE(MMMU_BASELINE_PIPELINE):
    # Test config the MMMU benchmark with MultipleChoiceTestModel and TestMMDataLoader
    def configure_pipeline(self, resume_from=None):
        config = super().configure_pipeline(model_config=ModelConfig(MultipleChoiceTestModel, {}))

        self.data_processing_comp.data_reader_config.init_args["split"] = "dev"
        self.data_processing_comp.data_reader_config.init_args["tasks"] = ["Math"]

        self.inference_comp.data_loader_config.class_name = TestMMDataLoader
        self.inference_comp.data_loader_config.init_args["n_iter"] = N_ITER
        return config


class TEST_DROP_PIPELINE(Drop_Experiment_Pipeline):
    def configure_pipeline(self):
        config = super().configure_pipeline(model_config=ModelConfig(GenericTestModel, {}))
        self.inference_comp.data_loader_config.class_name = TestDataLoader
        self.inference_comp.data_loader_config.init_args = {
            "path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl"),
            "n_iter": N_ITER,
        }
        return config


class TEST_AIME_PIPELINE(AIME_PIPELINE):
    # Test config the AIME benchmark with GenericTestModel and TestMMDataLoader
    def configure_pipeline(self):
        config = super().configure_pipeline(
            model_config=ModelConfig(GenericTestModel, {})
        )  # use a small subset of AIME
        self.inference_comp.data_loader_config.class_name = TestMMDataLoader
        self.inference_comp.data_loader_config.init_args["n_iter"] = N_ITER
        return config


class PipelineTest:
    def setUp(self) -> None:
        self.conf = self.get_config()
        logging.info(f"Pipeline test for {self.__class__.__name__}:")
        pipeline = Pipeline(self.conf.component_configs, self.conf.log_dir)
        self.eval_config = self.conf.component_configs[-1]
        self.data_reader_config = self.conf.component_configs[0]
        pipeline.run()
        # find all the files in the log directory recursively
        self.files = list(Path(self.conf.log_dir).rglob("*"))

    def test_outputs_exist(self) -> None:
        logging.info("Running test_outputs_exist test in PipelineTest")
        self.assertTrue(any("transformed_data.jsonl" in str(file) for file in self.files))
        if self.data_reader_config.prompt_template_path:
            self.assertTrue(any("processed_prompts.jsonl" in str(file) for file in self.files))
        self.assertTrue(any("inference_result.jsonl" in str(file) for file in self.files))
        if self.eval_config.metric_config is not None:
            self.assertTrue(any("metric_results.jsonl" in str(file) for file in self.files))
        n_aggregators = len(self.eval_config.aggregator_configs)
        n_aggregator_files = len([file for file in self.files if "aggregator" in str(file)])
        self.assertEqual(n_aggregators, n_aggregator_files)


@unittest.skipIf("skip_tests_with_missing_ds" in os.environ, "Missing public dataset. TODO: revert")
class SR1_PipelineTest(PipelineTest, unittest.TestCase):
    def get_config(self):
        return TEST_SPATIAL_REASONING_PIPELINE().pipeline_config


@unittest.skipIf("skip_tests_with_missing_ds" in os.environ, "Missing public dataset. TODO: revert")
class VP1_PipelineTest(PipelineTest, unittest.TestCase):
    def get_config(self):
        return TEST_VISUAL_PROMPTING_PIPELINE().pipeline_config


@unittest.skipIf("skip_tests_with_missing_ds" in os.environ, "Missing public dataset. TODO: revert")
class OR1_PipelineTest(PipelineTest, unittest.TestCase):
    def get_config(self):
        return TEST_OBJECT_RECOGNITION_PIPELINE().pipeline_config


@unittest.skipIf("skip_tests_with_missing_ds" in os.environ, "Missing public dataset. TODO: revert")
class OD1_PipelineTest(PipelineTest, unittest.TestCase):
    def get_config(self):
        return TEST_OBJECT_DETECTION_PIPELINE().pipeline_config


class MMMU_PipelineTest(PipelineTest, unittest.TestCase):
    def get_config(self):
        return TEST_MMMU_PIPELINE().pipeline_config


@unittest.skipIf("skip_tests_with_missing_ds" in os.environ, "Missing public dataset. TODO: revert")
class SPATIAL_GRID_PipelineTest(PipelineTest, unittest.TestCase):
    def get_config(self):
        return TEST_SPATIAL_GRID_PIPELINE().pipeline_config


@unittest.skipIf("skip_tests_with_missing_ds" in os.environ, "Missing public dataset. TODO: revert")
class SPATIAL_GRID_TEXTONLY_PipelineTest(PipelineTest, unittest.TestCase):
    def get_config(self):
        return TEST_SPATIAL_GRID_TEXTONLY_PIPELINE().pipeline_config


@unittest.skipIf("skip_tests_with_missing_ds" in os.environ, "Missing public dataset. TODO: revert")
class SPATIAL_MAP_PipelineTest(PipelineTest, unittest.TestCase):
    def get_config(self):
        return TEST_SPATIAL_MAP_PIPELINE().pipeline_config


@unittest.skipIf("skip_tests_with_missing_ds" in os.environ, "Missing public dataset. TODO: revert")
class SPATIAL_MAP_TEXTONLY_PipelineTest(PipelineTest, unittest.TestCase):
    def get_config(self):
        return TEST_SPATIAL_MAP_TEXTONLY_PIPELINE().pipeline_config


@unittest.skipIf("skip_tests_with_missing_ds" in os.environ, "Missing public dataset. TODO: revert")
class MAZE_PipelineTest(PipelineTest, unittest.TestCase):
    def get_config(self):
        return TEST_MAZE_PIPELINE().pipeline_config


@unittest.skipIf("skip_tests_with_missing_ds" in os.environ, "Missing public dataset. TODO: revert")
class MAZE_TEXTONLY_PipelineTest(PipelineTest, unittest.TestCase):
    def get_config(self):
        return TEST_MAZE_TEXTONLY_PIPELINE().pipeline_config


class GR1_PipelineTest(PipelineTest, unittest.TestCase):
    def get_config(self):
        return TEST_GEOMETRIC_REASONING_PIPELINE().pipeline_config


class DNA_PipelineTest(PipelineTest, unittest.TestCase):
    def get_config(self):
        return TEST_DNA_PIPELINE().pipeline_config

    def test_labels(self):
        logging.info("Running test_labels test in DNA_PipelineTest")
        eval_inference_output_file = os.path.join(self.conf.component_configs[-2].output_dir, "transformed_data.jsonl")
        with jsonlines.open(eval_inference_output_file, "r") as reader:
            eval_inference_data = [item for item in reader.iter(skip_empty=True, skip_invalid=True)]
            self.assertTrue(
                all(
                    item["model_action_label"] in range(0, 7) and item["model_harmless_label"] in range(0, 2)
                    for item in eval_inference_data
                )
            )


class IFEval_PipelineTest(PipelineTest, unittest.TestCase):
    def get_config(self):
        self.test_pipeline = TEST_IFEval_PIPELINE()
        self.config = self.test_pipeline.pipeline_config
        return self.config

    def setUp(self) -> None:
        super().setUp()
        self.eval_configs = [
            self.test_pipeline.evalreporting_comp,
            self.test_pipeline.instruction_level_evalreporting_comp,
        ]

    def test_outputs_exist(self) -> None:
        logging.info("Running test_outputs_exist test in PipelineTest")
        self.assertTrue(any("transformed_data.jsonl" in str(file) for file in self.files))
        if self.data_reader_config.prompt_template_path:
            self.assertTrue(any("processed_prompts.jsonl" in str(file) for file in self.files))
        self.assertTrue(any("inference_result.jsonl" in str(file) for file in self.files))
        if self.eval_config.metric_config is not None:
            self.assertTrue(any("metric_results.jsonl" in str(file) for file in self.files))
        n_aggregators = len([config for eval_config in self.eval_configs for config in eval_config.aggregator_configs])
        n_aggregator_files = len([file for file in self.files if "aggregator" in str(file)])
        self.assertEqual(n_aggregators, n_aggregator_files)


class TOXIGEN_PipelineTest(PipelineTest, unittest.TestCase):
    def get_config(self):
        return TEST_TOXIGEN_PIPELINE().pipeline_config


class KITAB_ONE_BOOK_CONSTRAINT_PIPELINE_PipelineTest(PipelineTest, unittest.TestCase):
    def get_config(self):
        return TEST_KITAB_ONE_BOOK_CONSTRAINT_PIPELINE().pipeline_config


class DROP_PipelineTest(PipelineTest, unittest.TestCase):
    def get_config(self):
        return TEST_DROP_PIPELINE().pipeline_config


class AIME_PipelineTest(PipelineTest, unittest.TestCase):
    def get_config(self):
        return TEST_AIME_PIPELINE().pipeline_config


if __name__ == "__main__":
    unittest.main()
