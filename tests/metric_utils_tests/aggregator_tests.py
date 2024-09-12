import os
import unittest

import numpy as np
import pandas as pd

from eureka_ml_insights.configs import create_logdir
from eureka_ml_insights.metrics import (
    Aggregator,
    AverageAggregator,
    AverageSTDDevAggregator,
    BiLevelAverageAggregator,
    BiLevelCountAggregator,
    CountAggregator,
    SumAggregator,
    TwoColumnSumAverageAggregator,
)

PRECISION = 3


class TestData:
    def setUp(self):
        self.maxDiff = None
        self.data = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, 6],
                "col2": [1, 2, 3, 1, 2, 3],
                "col3": ["a", "a", "a", "c", "c", "c"],
                "col4": ["x", "x", "y", "y", "y", "y"],
            }
        )
        self.output_dir = "output_dir"
        self.precision = PRECISION


class TestAggregator(TestData, unittest.TestCase):
    def test_aggregator(self):
        agg = Aggregator(["col1", "col2"], self.output_dir)
        self.assertEqual(agg.column_names, ["col1", "col2"])
        self.assertEqual(agg.output_dir, self.output_dir)
        self.assertRaises(NotImplementedError, agg.aggregate, self.data)

    def test_aggregator_input_validation(self):
        agg = Aggregator("col1", self.output_dir)
        self.assertRaises(ValueError, agg.aggregate, self.data)


class TestAverageAggregator(TestData, unittest.TestCase):
    def test_average_aggregator(self):
        avg_agg = AverageAggregator(["col1", "col2"], self.output_dir)
        avg_agg.aggregate(self.data)
        self.assertEqual(
            avg_agg.aggregated_result,
            {
                "col1": sum(self.data["col1"]) / len(self.data["col1"]),
                "col2": sum(self.data["col2"]) / len(self.data["col2"]),
            },
        )

    def test_average_aggregator_input_validation(self):
        avg_agg = AverageAggregator("col3", self.output_dir)
        self.assertRaises(ValueError, avg_agg.aggregate, self.data)

    def test_average_aggregator_group_by(self):
        avg_agg = AverageAggregator(["col1", "col2"], self.output_dir, group_by="col3")
        avg_agg.aggregate(self.data)
        self.assertEqual(avg_agg.aggregated_result, {"col1": {"a": 2, "c": 5}, "col2": {"a": 2, "c": 2}})

    def test_average_aggregator_group_by_multiple_columns(self):
        self.output_dir = create_logdir("AverageAggregatorTests")

        avg_agg = AverageAggregator(["col1", "col2"], self.output_dir, group_by=["col3", "col4"])
        avg_agg.aggregate(self.data)
        self.assertEqual(
            avg_agg.aggregated_result,
            {
                "col1": {("a_x"): 1.5, ("a_y"): 3, ("c_y"): 5},
                "col2": {("a_x"): 1.5, ("a_y"): 3, ("c_y"): 2},
            },
        )

        avg_agg.write_results()
        self.assertTrue(os.path.exists(avg_agg.output_file))


class TestCountAggregator(TestData, unittest.TestCase):
    def test_count_aggregator(self):
        count_agg = CountAggregator(["col1", "col2"], self.output_dir)
        count_agg.aggregate(self.data)
        self.assertEqual(
            count_agg.aggregated_result, {"col1": {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}, "col2": {1: 2, 2: 2, 3: 2}}
        )

    def test_count_aggregator_input_validation(self):
        count_agg = CountAggregator("col3", self.output_dir)
        self.assertRaises(ValueError, count_agg.aggregate, self.data)

    def test_count_aggregator_group_by(self):
        count_agg = CountAggregator(["col2"], self.output_dir, group_by="col3")
        count_agg.aggregate(self.data)
        self.assertEqual(count_agg.aggregated_result, {"col2": {"a": {1: 1, 2: 1, 3: 1}, "c": {1: 1, 2: 1, 3: 1}}})

    def test_count_aggregator_group_by_multiple_columns(self):
        self.output_dir = create_logdir("CountAggregatorTests")

        count_agg = CountAggregator(["col1", "col2"], self.output_dir, group_by=["col3", "col4"])
        count_agg.aggregate(self.data)
        self.assertEqual(count_agg.aggregated_result["col1"]["a_x"][1], 1)
        self.assertEqual(count_agg.aggregated_result["col1"]["a_y"][3], 1)
        self.assertEqual(count_agg.aggregated_result["col1"]["c_y"][6], 1)
        self.assertEqual(count_agg.aggregated_result["col2"]["a_x"][1], 1)
        self.assertEqual(count_agg.aggregated_result["col2"]["a_y"][3], 1)
        self.assertEqual(count_agg.aggregated_result["col2"]["c_y"][3], 1)

        count_agg.write_results()
        self.assertTrue(os.path.exists(count_agg.output_file))


class TestAverageSTDDevAggregator(TestData, unittest.TestCase):
    def test_average_stddev_aggregator(self):
        avg_stddev_agg = AverageSTDDevAggregator(["col1"], self.output_dir)
        avg_stddev_agg.aggregate(self.data)
        self.assertEqual(
            avg_stddev_agg.aggregated_result,
            {
                "col1": {
                    "average": round(self.data["col1"].mean(), self.precision),
                    "std_dev": round(self.data["col1"].std(), self.precision),
                }
            },
        )

    def test_average_stddev_aggregator_input_validation(self):
        avg_stddev_agg = AverageSTDDevAggregator("col3", self.output_dir)
        self.assertRaises(ValueError, avg_stddev_agg.aggregate, self.data)

    def test_average_stddev_aggregator_group_by(self):
        avg_stddev_agg = AverageSTDDevAggregator(["col1"], self.output_dir, group_by="col3")
        avg_stddev_agg.aggregate(self.data)
        self.assertAlmostEqual(
            avg_stddev_agg.aggregated_result,
            {"col1": {"a": {"average": 2, "std_dev": 1}, "c": {"average": 5, "std_dev": 1}}},
            places=self.precision,
        )

    def test_average_stddev_aggregator_group_by_multiple_columns(self):
        self.output_dir = create_logdir("AverageSTDDevAggregatorTests")

        avg_stddev_agg = AverageSTDDevAggregator(["col1"], self.output_dir, group_by=["col3", "col4"])
        avg_stddev_agg.aggregate(self.data)
        self.assertAlmostEqual(
            avg_stddev_agg.aggregated_result["col1"]["a_x"],
            {"average": 1.5, "std_dev": 0.707},
            places=self.precision,
        )
        self.assertAlmostEqual(
            avg_stddev_agg.aggregated_result["col1"]["c_y"],
            {"average": 5, "std_dev": 1},
            places=self.precision,
        )
        avg_stddev_agg.write_results()
        self.assertTrue(os.path.exists(avg_stddev_agg.output_file))


class BiLevelAggregatorTestData:
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "data_point_id": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                "data_repeat_id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                "numeric_metric": [5, 8, 2, 3, 6, 8, 3, 4, 5, 8, 4, 2],
                "categorical_metric": ["x", "y", "z", "z", "y", "y", "z", "y", "x", "y", "y", "x"],
                "group": ["a", "a", "b", "b", "a", "a", "b", "b", "a", "a", "b", "b"],
            }
        )
        self.output_dir = "output_dir"
        self.precision = PRECISION


class BiLevelAverageAggregatorTest(BiLevelAggregatorTestData, unittest.TestCase):
    def test_bilevel_average_aggregator(self):
        avg_agg = BiLevelAverageAggregator(
            ["numeric_metric"], first_groupby="data_point_id", second_groupby="group", output_dir=self.output_dir
        )
        avg_agg.aggregate(self.data)
        expected = [
            {
                "group": "a",
                "numeric_metric_mean": np.mean([np.mean([5, 6, 5]), np.mean([8, 8, 8])]),
                "numeric_metric_std": np.std([np.mean([5, 6, 5]), np.mean([8, 8, 8])], ddof=1),
            },
            {
                "group": "b",
                "numeric_metric_mean": np.mean([np.mean([2, 3, 4]), np.mean([3, 4, 2])]),
                "numeric_metric_std": np.std([np.mean([2, 3, 4]), np.mean([3, 4, 2])], ddof=1),
            },
        ]

        for i in range(len(avg_agg.aggregated_result)):
            self.assertAlmostEqual(
                avg_agg.aggregated_result[i]["numeric_metric_mean"],
                expected[i]["numeric_metric_mean"],
                places=self.precision,
            )
            self.assertAlmostEqual(
                avg_agg.aggregated_result[i]["numeric_metric_std"],
                expected[i]["numeric_metric_std"],
                places=self.precision,
            )

    def test_bilevel_average_aggregator_2(self):
        avg_agg = BiLevelAverageAggregator(
            ["numeric_metric"], first_groupby="data_repeat_id", second_groupby=None, output_dir=self.output_dir
        )
        avg_agg.aggregate(self.data)
        expected_first_level = [np.mean([5, 8, 2, 3]), np.mean([6, 8, 3, 4]), np.mean([5, 8, 4, 2])]
        expected = [
            {"numeric_metric": {"mean": np.mean(expected_first_level), "std": np.std(expected_first_level, ddof=1)}}
        ]
        self.assertEqual(avg_agg.aggregated_result, expected)


class BiLevelCountAggregatorTest(BiLevelAggregatorTestData, unittest.TestCase):
    def test_bilevel_count_aggregator(self):
        count_agg = BiLevelCountAggregator(
            ["categorical_metric"], first_groupby="data_point_id", second_groupby="group", output_dir=self.output_dir
        )
        count_agg.aggregate(self.data)
        expected_result = [
            {
                "group": "a",
                "categorical_metric": {
                    "x": {"mean": 1.0, "std": np.nan},
                    "y": {"mean": 2.0, "std": 1.4142135623730951},
                },
            },
            {
                "group": "b",
                "categorical_metric": {
                    "z": {"mean": 1.5, "std": 0.7071067811865476},
                    "y": {"mean": 1.0, "std": 0.0},
                    "x": {"mean": 0.5, "std": np.nan},
                },
            },
        ]
        for i in range(len(count_agg.aggregated_result)):
            for key in count_agg.aggregated_result[i]["categorical_metric"]:
                for agg in ["mean", "std"]:
                    actual = count_agg.aggregated_result[i]["categorical_metric"][key][agg]
                    expected = expected_result[i]["categorical_metric"][key][agg]
                    if np.isnan(expected):
                        self.assertTrue(np.isnan(actual))
                    else:
                        self.assertAlmostEqual(actual, expected, places=self.precision)

    def test_bilevel_count_aggregator_2(self):
        count_agg = BiLevelCountAggregator(
            ["categorical_metric"], first_groupby="data_repeat_id", second_groupby=None, output_dir=self.output_dir
        )
        count_agg.aggregate(self.data)
        expected = [
            {
                "categorical_metric": {
                    "x": {"mean": 1.0, "std": 0.707},
                    "y": {"mean": 2.0, "std": 1.0},
                    "z": {"mean": 1.0, "std": 0.707},
                }
            }
        ]
        for i in range(len(count_agg.aggregated_result)):
            self.assertAlmostEqual(
                count_agg.aggregated_result[i]["categorical_metric"]["x"]["mean"],
                expected[i]["categorical_metric"]["x"]["mean"],
                places=self.precision,
            )
            self.assertAlmostEqual(
                count_agg.aggregated_result[i]["categorical_metric"]["x"]["std"],
                expected[i]["categorical_metric"]["x"]["std"],
                places=self.precision,
            )


class TestTwoColumnSumAverageAggregator(TestData, unittest.TestCase):
    def test_average_aggregator(self):
        avg_agg = TwoColumnSumAverageAggregator("col1", "col2", self.output_dir)
        avg_agg.aggregate(self.data)
        self.assertEqual(
            avg_agg.aggregated_result,
            {"ratio": sum(self.data["col1"]) / sum(self.data["col2"])},
        )

    def test_average_aggregator_input_validation(self):
        avg_agg = TwoColumnSumAverageAggregator("col3", "col1", self.output_dir)
        self.assertRaises(ValueError, avg_agg.aggregate, self.data)

    def test_average_aggregator_group_by(self):
        avg_agg = TwoColumnSumAverageAggregator("col1", "col2", self.output_dir, group_by="col3")
        avg_agg.aggregate(self.data)
        self.assertEqual(avg_agg.aggregated_result, {"ratio": {"a": 1, "c": 2.5}})

    def test_average_aggregator_group_by_multiple_columns(self):
        self.output_dir = create_logdir("TwoColumnSumAverageAggregatorTests")

        avg_agg = TwoColumnSumAverageAggregator("col1", "col2", self.output_dir, group_by=["col3", "col4"])
        avg_agg.aggregate(self.data)
        self.assertEqual(
            avg_agg.aggregated_result,
            {"ratio": {("a_x"): 1, ("a_y"): 1, ("c_y"): 2.5}},
        )

        avg_agg.write_results()
        self.assertTrue(os.path.exists(avg_agg.output_file))


class TestSumAggregator(TestData, unittest.TestCase):
    def test_average_aggregator(self):
        avg_agg = SumAggregator(["col1"], self.output_dir)
        avg_agg.aggregate(self.data)
        self.assertEqual(avg_agg.aggregated_result, {"col1": sum(self.data["col1"])})

    def test_average_aggregator_input_validation(self):
        avg_agg = SumAggregator(["col3"], self.output_dir)
        self.assertRaises(ValueError, avg_agg.aggregate, self.data)

    def test_average_aggregator_group_by(self):
        avg_agg = SumAggregator(["col1"], self.output_dir, group_by="col3")
        avg_agg.aggregate(self.data)
        self.assertEqual(avg_agg.aggregated_result, {"col1": {"a": 6, "c": 15}})

    def test_average_aggregator_group_by_multiple_columns(self):
        self.output_dir = create_logdir("SumAggregatorTests")

        avg_agg = SumAggregator(["col1"], self.output_dir, group_by=["col3", "col4"])
        avg_agg.aggregate(self.data)
        self.assertEqual(avg_agg.aggregated_result, {"col1": {("a_x"): 3, ("a_y"): 3, ("c_y"): 15}})

        avg_agg.write_results()
        self.assertTrue(os.path.exists(avg_agg.output_file))


if __name__ == "__main__":
    unittest.main()
