import ast
import datetime
import json
import os

import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from eureka_ml_insights.data_utils import JsonReader


class Aggregator:
    """Aggregates data and writes the results."""

    def __init__(self, column_names, output_dir, group_by=None, ignore_non_numeric=False, filename_base=None, **kwargs):
        """
        Initializes an Aggregator.

        Args:
            column_names (List[str]): A list of column names to aggregate.
            output_dir (str): The directory to save the report.
            group_by (Union[str, List[str]], optional): The column(s) to group by before aggregating. Defaults to None.
            ignore_non_numeric (bool, optional): If True, ignores non-numeric values for the average aggregator.
                Defaults to False.
            filename_base (str, optional): An optional base string to be used in the file name for the report. If not
                None, the report filename will concatenate the class name, datetime, and filename_base. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.column_names = column_names
        self.group_by = group_by
        self.output_dir = output_dir
        self.aggregated_result = None
        self.ignore_non_numeric = ignore_non_numeric
        self.filename_base = filename_base

    def aggregate(self, data):
        """
        Aggregates the provided data.

        Args:
            data (pd.DataFrame): The input data to aggregate.
        """
        if self.ignore_non_numeric:
            data = data[data["is_valid"]].copy()
        self._validate_data(data)
        if self.group_by:
            if isinstance(self.group_by, list):
                group_name = "_".join(self.group_by)
                data[group_name] = data[self.group_by].astype(str).agg("_".join, axis=1)
                self.group_by = group_name
            self._aggregate_grouped(data)
        else:
            self._aggregate(data)

    def __str__(self):
        """
        Returns a string representation of the aggregator, used for naming output files.

        Returns:
            str: The string representation, including date/time and optionally column/group info.
        """
        str_rep = self.__class__.__name__.lower()
        str_rep += "_" + datetime.datetime.now().strftime("%H%M%S%f")
        if self.filename_base is None:
            if self.column_names:
                str_rep = str_rep + "_on_" + "_".join(self.column_names)
            if self.group_by:
                str_rep += "_grouped_by_" + str(self.group_by)
        else:
            str_rep += "_" + self.filename_base
        return str_rep

    def write_results(self):
        """
        Writes the aggregated results to a JSON file.

        Raises:
            ValueError: If the data has not been aggregated yet.
        """
        if self.aggregated_result is None:
            raise ValueError("The data has not been aggregated yet.")

        self.output_file = os.path.join(self.output_dir, str(self) + "_report.json")
        with open(self.output_file, "w") as f:
            json.dump(self.aggregated_result, f)

    def _validate_data(self, data, **kwargs):
        """
        Ensures that the input arguments are in the correct format.

        Args:
            data (pd.DataFrame): The input data to validate.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If column_names is not a list of strings, or if group_by is not a string or list of strings.
        """
        if not isinstance(self.column_names, list) or not all(isinstance(col, str) for col in self.column_names):
            raise ValueError("column_names must be a list of strings.")
        if self.group_by:
            if not isinstance(self.group_by, str):
                if not isinstance(self.group_by, list) or not all(isinstance(col, str) for col in self.group_by):
                    raise ValueError("group_by must be a string or a list of strings")

    def _aggregate(self, data):
        """
        Aggregates the data without grouping.

        Args:
            data (pd.DataFrame): The input data to aggregate.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _aggregate_grouped(self, data):
        """
        Aggregates the data with grouping.

        Args:
            data (pd.DataFrame): The input data to aggregate.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


class NumericalAggregator(Aggregator):
    """Base class for aggregators that require the data to be numeric."""

    def _validate_data(self, data):
        """
        Ensures that the data is numeric, in addition to the validations in the superclass.

        Args:
            data (pd.DataFrame): The input data to validate.

        Raises:
            ValueError: If the data fails any numeric check or if column_names/group_by are not valid.
        """
        super()._validate_data(data)
        for col in self.column_names:
            data[col] = pd.to_numeric(data[col], errors="raise")


class SumAggregator(NumericalAggregator):
    """Aggregates data by summing the values."""

    def _aggregate(self, data):
        """
        Aggregates data without grouping by computing sums for each specified column.

        Args:
            data (pd.DataFrame): The input data to aggregate.
        """
        sums = {col: data[col].sum() for col in self.column_names}
        self.aggregated_result = sums

    def _aggregate_grouped(self, data):
        """
        Aggregates data with grouping by computing sums for each specified column.

        Args:
            data (pd.DataFrame): The input data to aggregate.
        """
        gb = data.groupby(self.group_by)
        sums = {col: gb[col].sum().to_dict() for col in self.column_names}
        self.aggregated_result = sums


class MaxAggregator(NumericalAggregator):
    """Aggregates data by taking the maximum of the values."""

    def _aggregate(self, data):
        """
        Aggregates data without grouping by computing the maximum value for each specified column.

        Args:
            data (pd.DataFrame): The input data to aggregate.
        """
        maxes = {col: data[col].max() for col in self.column_names}
        self.aggregated_result = maxes

    def _aggregate_grouped(self, data):
        """
        Aggregates data with grouping by computing the maximum value for each specified column.

        Args:
            data (pd.DataFrame): The input data to aggregate.
        """
        gb = data.groupby(self.group_by)
        maxes = {col: gb[col].max().to_dict() for col in self.column_names}
        self.aggregated_result = maxes


class AverageAggregator(NumericalAggregator):
    """Aggregates data by computing the average of numeric columns."""

    def _aggregate(self, data):
        """
        Aggregates data without grouping by computing the mean (rounded to 3 decimals) for each specified column.

        Args:
            data (pd.DataFrame): The input data to aggregate.
        """
        if len(data) == 0:
            averages = {col: 0 for col in self.column_names}
        else:
            averages = {col: round(data[col].mean(), 3) for col in self.column_names}
        self.aggregated_result = averages

    def _aggregate_grouped(self, data):
        """
        Aggregates data with grouping by computing the mean (rounded to 3 decimals) for each specified column.

        Args:
            data (pd.DataFrame): The input data to aggregate.
        """
        if len(data) == 0:
            averages = {col: 0 for col in self.column_names}
        else:
            gb = data.groupby(self.group_by)
            averages = {col: round(gb[col].mean(), 3).to_dict() for col in self.column_names}
        self.aggregated_result = averages


class AverageSTDDevAggregator(NumericalAggregator):
    """Aggregates data by computing both the average and standard deviation for numeric columns."""

    def _aggregate(self, data):
        """
        Aggregates data without grouping by computing the mean and standard deviation of specified columns.

        Args:
            data (pd.DataFrame): The input data to aggregate.
        """
        averages = {col: round(data[col].mean(), 3) for col in self.column_names}
        std_devs = {col: round(data[col].std(), 3) for col in self.column_names}
        self.aggregated_result = {
            col: {"average": averages[col], "std_dev": std_devs[col]} for col in self.column_names
        }

    def _aggregate_grouped(self, data):
        """
        Aggregates data with grouping by computing the mean and standard deviation of specified columns.

        Args:
            data (pd.DataFrame): The input data to aggregate.
        """
        gb = data.groupby(self.group_by)
        averages = {col: gb[col].mean().round(3).to_dict() for col in self.column_names}
        std_devs = {col: gb[col].std().round(3).to_dict() for col in self.column_names}
        self.aggregated_result = {
            col: {group: {"average": averages[col][group], "std_dev": std_devs[col][group]} for group in averages[col]}
            for col in self.column_names
        }


class CountAggregator(Aggregator):
    """Counts the occurrences of values in columns and can optionally normalize the counts."""

    def __init__(self, column_names, output_dir, group_by=None, normalize=False, **kwargs):
        """
        Initializes a CountAggregator.

        Args:
            column_names (List[str]): A list of column names to count.
            output_dir (str): The directory to save the report.
            group_by (Union[str, List[str]], optional): The column(s) to group by before aggregating. Defaults to None.
            normalize (bool, optional): If True, normalizes the counts to be between 0 and 1. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(column_names, output_dir, group_by, **kwargs)
        self.normalize = normalize

    def __str__(self):
        """
        Returns a string representation of the aggregator, used for naming output files.
        If normalization is used, '_normalized' is appended to the name.

        Returns:
            str: The string representation, including date/time and optionally normalization info.
        """
        str_rep = super().__str__()
        if self.normalize:
            str_rep += "_normalized"
        return str_rep

    def _aggregate(self, data):
        """
        Aggregates data without grouping by counting occurrences of values in each specified column.

        Args:
            data (pd.DataFrame): The input data to aggregate.
        """
        counts = {col: data[col].value_counts(normalize=self.normalize).round(3).to_dict() for col in self.column_names}
        self.aggregated_result = counts

    def _aggregate_grouped(self, data):
        """
        Aggregates data with grouping by counting occurrences of values in each specified column.

        Args:
            data (pd.DataFrame): The input data to aggregate.
        """
        gb = data.groupby(self.group_by)
        col_counts = {
            col: gb[col].value_counts(normalize=self.normalize).unstack(level=0).round(3).to_dict()
            for col in self.column_names
        }
        self.aggregated_result = col_counts


class BiLevelAggregator(AverageAggregator):
    """
    Aggregates the data in two levels. First, it groups the data by the first_groupby column(s) and applies an
    aggregation function (agg_fn) on the specified columns. Then, if a second_groupby is specified, it groups
    the result again and computes the mean and standard deviation.
    """

    def __init__(self, column_names, first_groupby, output_dir, second_groupby=None, agg_fn="mean", **kwargs):
        """
        Initializes a BiLevelAggregator.

        Args:
            column_names (List[str]): A list of column names to aggregate.
            first_groupby (Union[str, List[str]]): The column(s) for the first level of grouping.
            output_dir (str): The directory to save the report.
            second_groupby (Union[str, List[str]], optional): The column(s) for the second level of grouping.
                Defaults to None.
            agg_fn (str, optional): The aggregation function to apply for the first grouping. Defaults to "mean".
            **kwargs: Additional keyword arguments.
        """
        super().__init__(column_names, output_dir, group_by=None, **kwargs)
        self.first_groupby = first_groupby
        self.second_groupby = second_groupby
        self.agg_fn = agg_fn

    def _aggregate(self, data):
        """
        Performs a two-level aggregation. First, it groups by the first_groupby column(s) and applies agg_fn to
        the specified columns. Then, if second_groupby is provided, groups the intermediate result again and
        computes mean and standard deviation. If second_groupby is not provided, only mean and standard deviation
        of the first aggregation are computed.

        Args:
            data (pd.DataFrame): The input data to aggregate.
        """
        gb = data.groupby(self.first_groupby)
        agg_map = {col: self.agg_fn for col in self.column_names}
        agg_map.update(
            {
                col: "first"
                for col in data.columns
                if col not in self.column_names and col != self.first_groupby and col not in self.first_groupby
            }
        )

        first_result = gb.aggregate(agg_map).reset_index()
        if self.second_groupby:
            gb = first_result.groupby(self.second_groupby)
            agg_map = {col: ["mean", "std"] for col in self.column_names}
            second_result = gb.agg(agg_map).reset_index()
            second_result.columns = [f"{col}_{agg}" if agg else col for col, agg in second_result.columns]
            self.aggregated_result = second_result.to_dict(orient="records")
        else:
            self.aggregated_result = []
            for col in self.column_names:
                col_mean = first_result[col].mean()
                col_std = first_result[col].std()
                self.aggregated_result.append({col: {"mean": col_mean, "std": col_std}})


class BiLevelCountAggregator(Aggregator):
    """
    Aggregates the data in two levels. First, it groups the data by the first_groupby column(s) and
    applies a value_counts aggregation to each of the specified columns. Then, if a second_groupby is given,
    it groups the intermediate result again by the second_groupby column(s) and computes the mean and standard
    deviation of the counts.
    """

    def __init__(self, column_names, first_groupby, output_dir, normalize=False, second_groupby=None, **kwargs):
        """
        Initializes a BiLevelCountAggregator.

        Args:
            column_names (List[str]): A list of column names to aggregate.
            first_groupby (Union[str, List[str]]): The column(s) for the first level of grouping.
            output_dir (str): The directory to save the report.
            normalize (bool, optional): If True, normalizes the counts to be between 0 and 1. Defaults to False.
            second_groupby (Union[str, List[str]], optional): The column(s) for the second level of grouping.
                Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(column_names, output_dir, group_by=first_groupby, **kwargs)
        self.first_groupby = first_groupby
        self.second_groupby = second_groupby
        self.normalize = normalize
        self.agg_fn = lambda x: x.value_counts(normalize=self.normalize).to_dict()

    def _aggregate_grouped(self, data):
        """
        Performs the two-level aggregation: first by the first_groupby, then optionally by the second_groupby.
        Aggregates using value_counts for each column.

        Args:
            data (pd.DataFrame): The input data to aggregate.
        """
        gb = data.groupby(self.first_groupby)
        agg_map = {col: self.agg_fn for col in self.column_names}
        agg_map.update(
            {
                col: "first"
                for col in data.columns
                if col not in self.column_names and col != self.first_groupby and col not in self.first_groupby
            }
        )

        first_result = gb.aggregate(agg_map).reset_index()

        def agg_counts_by_avg_std(x):
            counts = {}
            for row in x:
                for key, value in row.items():
                    if key not in counts:
                        counts[key] = []
                    counts[key].append(value)
            return {key: {"mean": sum(value) / len(x), "std": pd.Series(value).std()} for key, value in counts.items()}

        if self.second_groupby:
            gb = first_result.groupby(self.second_groupby)
            second_result = gb.agg({col: agg_counts_by_avg_std for col in self.column_names}).reset_index()
            self.aggregated_result = second_result.to_dict(orient="records")
        else:
            self.aggregated_result = []
            for col in self.column_names:
                self.aggregated_result.append({col: agg_counts_by_avg_std(first_result[col])})


class TwoColumnSumAverageAggregator(NumericalAggregator):
    """
    Aggregates data from two columns by summing the values in each column and then
    dividing the sum of the numerator column by the sum of the denominator column.
    """

    def __init__(self, numerator_column_name, denominator_column_name, output_dir, group_by=None, **kwargs):
        """
        Initializes a TwoColumnSumAverageAggregator.

        Args:
            numerator_column_name (str): The name of the column containing the numerator values.
            denominator_column_name (str): The name of the column containing the denominator values.
            output_dir (str): The directory where the aggregated result will be stored.
            group_by (Union[str, List[str]], optional): The column(s) to group the data by. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__([numerator_column_name, denominator_column_name], output_dir, group_by=group_by, **kwargs)
        self.numerator_column_name = numerator_column_name
        self.denominator_column_name = denominator_column_name

    def _aggregate(self, data):
        """
        Aggregates data without grouping by computing the sum of both columns and the
        ratio of sum(numerator) to sum(denominator).

        Args:
            data (pd.DataFrame): The input data to aggregate.
        """
        sums = {col: data[col].sum() for col in self.column_names}
        divided_result = sums[self.numerator_column_name] / sums[self.denominator_column_name]
        self.aggregated_result = {"ratio": divided_result}

    def _aggregate_grouped(self, data):
        """
        Aggregates data with grouping by computing the sum of both columns within each group
        and the ratio of sum(numerator) to sum(denominator).

        Args:
            data (pd.DataFrame): The input data to aggregate.
        """
        gb = data.groupby(self.group_by)
        divided_result = (gb[self.numerator_column_name].sum() / gb[self.denominator_column_name].sum()).to_dict()
        self.aggregated_result = {"ratio": divided_result}


class ValueFilteredAggregator(Aggregator):
    """
    Aggregator that filters out a particular value from the specified columns before aggregating the data
    using a provided aggregator class.
    """

    def __init__(
        self,
        agg_class,
        value,
        column_names,
        output_dir,
        group_by=None,
        ignore_non_numeric=False,
        filename_base=None,
        **kwargs,
    ):
        """
        Initializes a ValueFilteredAggregator.

        Args:
            agg_class (class): The aggregator class to use for aggregation.
            value (Any): The value to filter out.
            column_names (List[str]): The column names to filter and aggregate.
            output_dir (str): The directory to save the report.
            group_by (Union[str, List[str]], optional): The column(s) to group by before aggregating. Defaults to None.
            ignore_non_numeric (bool, optional): If True, ignores non-numeric values for the average aggregator.
                Defaults to False.
            filename_base (str, optional): An optional base string to be used in the file name for the report. If not
                None, the report filename will concatenate the class name, datetime, and filename_base. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.base_aggregator = agg_class(
            column_names, output_dir, group_by, ignore_non_numeric, filename_base, **kwargs
        )
        self.value = value
        self.column_names = column_names
        self.group_by = group_by
        self.output_dir = output_dir
        self.aggregated_result = None
        self.ignore_non_numeric = ignore_non_numeric
        self.filename_base = filename_base

    def aggregate(self, data):
        """
        Filters out the specified value in each column before calling the base aggregator.

        Args:
            data (pd.DataFrame): The input data to aggregate.
        """
        agg_results = {}
        for col in self.column_names:
            filtered_data = data[data[col] != self.value].copy()
            self.base_aggregator.column_names = [col]
            self.base_aggregator.aggregate(filtered_data)
            agg_results.update(self.base_aggregator.aggregated_result)
        self.aggregated_result = agg_results


class CocoDetectionAggregator(Aggregator):
    """Uses the COCO tools to calculate AP50 for provided detections."""

    def __init__(self, column_names, output_dir, target_coco_json_reader: JsonReader):
        """
        Initializes a CocoDetectionAggregator.

        Args:
            column_names (List[str]): A single column name (in a list) containing detection results.
            output_dir (str): The directory to save the reports.
            target_coco_json_reader (JsonReader): A reader to load the ground-truth JSON for detections (in COCO format).
        """
        super().__init__(column_names, output_dir)

        self.coco = COCO()
        self.coco.dataset = target_coco_json_reader.read()
        self.coco.createIndex()

    def _aggregate(self, data):
        """
        Aggregates detections by computing the AP50 metric using the COCOeval library.

        Args:
            data (pd.DataFrame): The input data containing detection annotations in the specified column.
        """
        annotations = []
        for ann in data[self.column_names[0]].tolist():
            if ann != "none":
                annotations += ast.literal_eval(ann)

        if annotations:
            cocovalPrediction = self.coco.loadRes(annotations)
            cocoEval = COCOeval(self.coco, cocovalPrediction, "bbox")
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            self.aggregated_result = [{self.column_names[0]: {"AP50": cocoEval.stats[1]}}]


class Reporter:
    """Applies various aggregations and visualizations to the data."""

    def __init__(self, output_dir, aggregator_configs=None, visualizer_configs=None):
        """
        Initializes a Reporter.

        Args:
            output_dir (str): The directory to save the reports.
            aggregator_configs (List[object], optional): A list of AggregatorConfig objects. Defaults to None.
            visualizer_configs (List[object], optional): A list of VisualizerConfig objects. Defaults to None.
        """
        self.aggregators = (
            [config.class_name(**dict(output_dir=output_dir, **config.init_args)) for config in aggregator_configs]
            if aggregator_configs
            else []
        )
        self.visualizers = (
            [config.class_name(**dict(output_dir=output_dir, **config.init_args)) for config in visualizer_configs]
            if visualizer_configs
            else []
        )
        self.output_dir = output_dir

    def generate_report(self, data):
        """
        Generates a report by applying configured aggregators and visualizers to the data.

        Args:
            data (pd.DataFrame): The input data to generate a report for.
        """
        for aggregator in self.aggregators:
            aggregator.aggregate(data)
            aggregator.write_results()
        for visualizer in self.visualizers:
            visualizer.visualize(data)
