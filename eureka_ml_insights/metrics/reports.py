import ast
import datetime
import json
import os

import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from eureka_ml_insights.data_utils import JsonReader


class Aggregator:
    """This class aggregates data and writes the results."""

    def __init__(self, column_names, output_dir, group_by=None, ignore_non_numeric=False, filename_base=None, **kwargs):
        """
        args:
            column_names: list of column names to aggregate
            output_dir: str. directory to save the report
            group_by: str. or list of str. column(s) to group by before aggregating
            ignore_non_numeric: bool. if True ignore non-numeric values for average aggregator
            filename_base: str. optional base string to be used in the file name for the report. If not None, the report filename will concatenate the class name, datetime, and filename_base.
        """

        self.column_names = column_names
        self.group_by = group_by
        self.output_dir = output_dir
        self.aggregated_result = None
        self.ignore_non_numeric = ignore_non_numeric
        self.filename_base = filename_base

    def aggregate(self, data):
        if self.ignore_non_numeric:
            data = data[data["is_valid"]].copy()
        # determine if a groupby is needed, and call the appropriate aggregation function
        self._validate_data(data)
        if self.group_by:
            # if group_by is a list, create a new column that is concatenation of the str values
            if isinstance(self.group_by, list):
                group_name = "_".join(self.group_by)
                data[group_name] = data[self.group_by].astype(str).agg("_".join, axis=1)
                self.group_by = group_name
            self._aggregate_grouped(data)
        else:
            self._aggregate(data)

    def __str__(self):
        str_rep = self.__class__.__name__.lower()
        # get time of day string down to miliseconds for unique name in case several of
        # the same aggregator are used in the same report
        str_rep += "_" + datetime.datetime.now().strftime("%H%M%S%f")
        # if filename_base is not provided during report creation, the report filename will concatenate the names of metric colums, as well as the "_grouped_by_" indicator if applicable.
        # if filename_base is provided during report creation, the report filename will concatenate the class name, datetime, and filename_base.
        if self.filename_base is None:
            if self.column_names:
                str_rep = str_rep + "_on_" + "_".join(self.column_names)
            if self.group_by:
                str_rep += "_grouped_by_" + str(self.group_by)
        else:
            str_rep += "_" + self.filename_base
        return str_rep

    def write_results(self):
        if self.aggregated_result is None:
            raise ValueError("The data has not been aggregated yet.")

        self.output_file = os.path.join(self.output_dir, str(self) + "_report.json")
        with open(self.output_file, "w") as f:
            json.dump(self.aggregated_result, f)

    def _validate_data(self, data, **kwargs):
        """Ensure that the input arguments are in the correct format."""
        if not isinstance(self.column_names, list) or not all(isinstance(col, str) for col in self.column_names):
            raise ValueError("column_names must be a list of strings.")
        if self.group_by:
            if not isinstance(self.group_by, str):
                if not isinstance(self.group_by, list) or not all(isinstance(col, str) for col in self.group_by):
                    raise ValueError("group_by must be a string or a list of strings")

    def _aggregate(self, data):
        """Aggregate the data without grouping."""
        raise NotImplementedError

    def _aggregate_grouped(self, data):
        """Aggregate the data with grouping."""
        raise NotImplementedError


class NumericalAggregator(Aggregator):
    """This class is a base class for aggregators that require the data to be numeric."""

    def _validate_data(self, data):
        super()._validate_data(data)
        """ Ensure that the data is numeric."""
        for col in self.column_names:
            data[col] = pd.to_numeric(data[col], errors="raise")


class SumAggregator(NumericalAggregator):
    """
    This class aggregates data by summing the values."""

    def _aggregate(self, data):
        sums = {col: data[col].sum() for col in self.column_names}
        self.aggregated_result = sums

    def _aggregate_grouped(self, data):
        gb = data.groupby(self.group_by)
        sums = {col: gb[col].sum().to_dict() for col in self.column_names}
        self.aggregated_result = sums


class MaxAggregator(NumericalAggregator):
    """
    This class aggregates data by taking the max of the values."""

    def _aggregate(self, data):
        maxes = {col: data[col].max() for col in self.column_names}
        self.aggregated_result = maxes

    def _aggregate_grouped(self, data):
        gb = data.groupby(self.group_by)
        maxes = {col: gb[col].max().to_dict() for col in self.column_names}
        self.aggregated_result = maxes


class AverageAggregator(NumericalAggregator):

    def _aggregate(self, data):
        if len(data) == 0:
            averages = {col: 0 for col in self.column_names}
        else:
            averages = {col: data[col].mean().round(3) for col in self.column_names}
        self.aggregated_result = averages

    def _aggregate_grouped(self, data):
        if len(data) == 0:
            averages = {col: 0 for col in self.column_names}
        else:
            gb = data.groupby(self.group_by)
            averages = {col: round(gb[col].mean(), 3).to_dict() for col in self.column_names}
        self.aggregated_result = averages


class AverageSTDDevAggregator(NumericalAggregator):

    def _aggregate(self, data):
        averages = {col: round(data[col].mean(), 3) for col in self.column_names}
        std_devs = {col: round(data[col].std(), 3) for col in self.column_names}
        self.aggregated_result = {
            col: {"average": averages[col], "std_dev": std_devs[col]} for col in self.column_names
        }

    def _aggregate_grouped(self, data):
        gb = data.groupby(self.group_by)
        averages = {col: gb[col].mean().round(3).to_dict() for col in self.column_names}
        std_devs = {col: gb[col].std().round(3).to_dict() for col in self.column_names}
        self.aggregated_result = {
            col: {group: {"average": averages[col][group], "std_dev": std_devs[col][group]} for group in averages[col]}
            for col in self.column_names
        }


class CountAggregator(Aggregator):
    """Counts the number of occurences of values in the columns and optionally normalize the counts."""

    def __init__(self, column_names, output_dir, group_by=None, normalize=False, **kwargs):
        """
        args:
            column_names: list of column names to aggregate
            output_dir: str. directory to save the report
            group_by: str. or list of str. column(s) to group by before aggregating
            normalize: bool. If True, normalize the counts to be between 0 and 1.
        """
        super().__init__(column_names, output_dir, group_by, **kwargs)
        self.normalize = normalize

    def __str__(self):
        str_rep = super().__str__()
        if self.normalize:
            str_rep += "_normalized"
        return str_rep

    def _aggregate(self, data):
        counts = {col: data[col].value_counts(normalize=self.normalize).round(3).to_dict() for col in self.column_names}
        self.aggregated_result = counts

    def _aggregate_grouped(self, data):
        # for each column, create a dictionary that contains the counts for each group
        gb = data.groupby(self.group_by)
        col_counts = {
            col: gb[col].value_counts(normalize=self.normalize).unstack(level=0).fillna(0).round(3).to_dict()
            for col in self.column_names
        }
        self.aggregated_result = col_counts


class BiLevelAggregator(AverageAggregator):
    """
    This class aggregates the data in two levels. It first groups the data by the first_groupby column and
    aggregates the data by applying the agg_fn on the column_names. It It then groups the result by the
    second_groupby column and aggregates again by taking the mean and standard deviation of
    the column_names.
    """

    def __init__(self, column_names, first_groupby, output_dir, second_groupby=None, agg_fn="mean", **kwargs):
        super().__init__(column_names, output_dir, group_by=None, **kwargs)
        self.first_groupby = first_groupby
        self.second_groupby = second_groupby
        self.agg_fn = agg_fn

    def _aggregate(self, data):
        # take the self.agg_fn aggregation of the column for each group in the first groupby,
        # aggregate the rest of the columns by 'first'
        gb = data.groupby(self.first_groupby)
        agg_map = {col: self.agg_fn for col in self.column_names}  # aggregate the column_names by self.agg_fn
        agg_map.update(
            {
                col: "first"
                for col in data.columns
                if col not in self.column_names  # aggregate the un-interesting columns by 'first'
                and col != self.first_groupby  # in case first_groupby is a single column
                and col not in self.first_groupby
            }  # in case there are multiple columns in the first_groupby
        )

        first_result = gb.aggregate(agg_map).reset_index()
        if self.second_groupby:
            # take the average and std of the first level aggregation for each group in the second groupby
            gb = first_result.groupby(self.second_groupby)
            agg_map = {col: ["mean", "std"] for col in self.column_names}
            # flatten the multi-level column index
            second_result = gb.agg(agg_map).reset_index()
            second_result.columns = [f"{col}_{agg}" if agg else col for col, agg in second_result.columns]
            self.aggregated_result = second_result.to_dict(orient="records")
        else:
            # take the average and std of the first level aggregation
            self.aggregated_result = []
            for col in self.column_names:
                col_mean = first_result[col].mean()
                col_std = first_result[col].std()
                self.aggregated_result.append({col: {"mean": col_mean, "std": col_std}})


class BiLevelCountAggregator(Aggregator):
    """
    This class aggregates the data in two levels. It first groups the data by the first_groupby column and
    aggregates the data by applying value_counts to each of the column_names. It then groups the result by the
    second_groupby column and aggregates it again by taking the mean and standard deviation of the counts over
    the groups.
    """

    def __init__(self, column_names, first_groupby, output_dir, normalize=False, second_groupby=None, **kwargs):
        super().__init__(column_names, output_dir, group_by=first_groupby, **kwargs)
        self.first_groupby = first_groupby
        self.second_groupby = second_groupby
        self.normalize = normalize
        self.agg_fn = lambda x: x.value_counts(normalize=self.normalize).to_dict()

    def _aggregate_grouped(self, data):
        # Count values in the columns for each group in the first groupby,
        # aggregate the rest of the columns by 'first'
        gb = data.groupby(self.first_groupby)
        agg_map = {col: self.agg_fn for col in self.column_names}  # aggregate the column_names by self.agg_fn
        agg_map.update(
            {
                col: "first"
                for col in data.columns
                if col not in self.column_names  # aggregate the un-interesting columns by 'first'
                and col != self.first_groupby  # in case first_groupby is a single column
                and col not in self.first_groupby
            }  # in case there are multiple columns in the first_groupby
        )

        first_result = gb.aggregate(agg_map).reset_index()

        # take the average and std of the first level aggregation for each group in the second groupby
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
    """This class that aggregates data from two columns by summing the values in each column and
    then dividing the sum of the numerator column by the sum of the denominator column.
    """

    def __init__(self, numerator_column_name, denominator_column_name, output_dir, group_by=None, **kwargs):
        """
        args:
            - numerator_column_name (str): The name of the column containing the numerator values.
            - denominator_column_name (str): The name of the column containing the denominator values.
            - output_dir (str): The directory where the aggregated result will be stored.
            - groupby (str, optional): The column name to group the data by. Defaults to None.
        """
        super().__init__([numerator_column_name, denominator_column_name], output_dir, group_by=group_by, **kwargs)
        self.numerator_column_name = numerator_column_name
        self.denominator_column_name = denominator_column_name

    def _aggregate(self, data):
        sums = {col: data[col].sum() for col in self.column_names}
        divided_result = sums[self.numerator_column_name] / sums[self.denominator_column_name]
        self.aggregated_result = {"ratio": divided_result}

    def _aggregate_grouped(self, data):
        gb = data.groupby(self.group_by)
        divided_result = (gb[self.numerator_column_name].sum() / gb[self.denominator_column_name].sum()).to_dict()
        self.aggregated_result = {"ratio": divided_result}


class ValueFilteredAggregator(Aggregator):
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
        Aggregator that filters out a particular value before aggregating the data.
        args:
            agg_class: Aggregator class to use for aggregation
            value: value to filter out
            column_names: column names to filter and aggregate
            output_dir: str. directory to save the report
            group_by: str. or list of str. column(s) to group by before aggregating
            ignore_non_numeric: bool. if True ignore non-numeric values for average aggregator
            filename_base: str. optional base string to be used in the file name for the report. If not None, the report filename will concatenate the class name, datetime, and filename_base.
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
        agg_results = {}
        for col in self.column_names:
            # workaround to process one column at a time
            filtered_data = data[data[col] != self.value].copy()
            self.base_aggregator.column_names = [col]
            self.base_aggregator.aggregate(filtered_data)
            agg_results.update(self.base_aggregator.aggregated_result)
        self.aggregated_result = agg_results


class CocoDetectionAggregator(Aggregator):
    """This class uses the coco tools to calculated AP50 for the provided detections."""

    def __init__(self, column_names, output_dir, target_coco_json_reader: JsonReader):
        """
        args:
            column_names: Single column (pass as a list to conform with superclass), that indicates which column has the detection results
            output_dir: str, directory to save the reports
            target_coco_json_reader: JsonReader, reader to load the ground truth json for the detections (in coco json format)
        """
        super().__init__(column_names, output_dir)

        self.coco = COCO()
        self.coco.dataset = target_coco_json_reader.read()
        self.coco.createIndex()

    def _aggregate(self, data):

        # pull the annotasions out of the data and form for the COCOeval library
        annotations = []
        for ann in data[self.column_names[0]].tolist():
            if ann != "none":
                annotations += ast.literal_eval(ann)

        # run the COCOeval library to get stats
        if annotations:
            cocovalPrediction = self.coco.loadRes(annotations)

            cocoEval = COCOeval(self.coco, cocovalPrediction, "bbox")
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            self.aggregated_result = [{self.column_names[0]: {"AP50": cocoEval.stats[1]}}]


class Reporter:
    """This class applies various aggregations and visualizations to the data."""

    def __init__(self, output_dir, aggregator_configs=None, visualizer_configs=None):
        """
        args:
            output_dir: str. directory to save the reports
            aggregator_configs: list of AggregatorConfig objects
            visualizer_configs: list of VisualizerConfig objects
        """
        self.aggregators = [
            config.class_name(**dict(output_dir=output_dir, **config.init_args)) for config in aggregator_configs
        ]
        self.visualizers = [
            config.class_name(**dict(output_dir=output_dir, **config.init_args)) for config in visualizer_configs
        ]
        self.output_dir = output_dir

    def generate_report(self, data):
        for aggregator in self.aggregators:
            aggregator.aggregate(data)
            aggregator.write_results()
        for visualizer in self.visualizers:
            visualizer.visualize(data)
