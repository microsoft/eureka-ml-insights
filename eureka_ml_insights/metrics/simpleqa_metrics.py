import re
from eureka_ml_insights.metrics.metrics_base import CompositeMetric
from eureka_ml_insights.metrics.reports import NumericalAggregator

class SimpleQA_Metric(CompositeMetric):
    """
    Composite metric for evaluating SimpleQA.
    """

    def __init__(self):
        super().__init__()

    def __evaluate__(self, row):
        return self.process_row(row)

    def process_row(self, row):
        grading_response = row["model_output"]
        match = re.search(r"(A|B|C)", grading_response)
        grade_letter = match.group(0) if match else "C"  # Default to "NOT_ATTEMPTED" if no match
        
        # Metrics based on grading response
        is_correct = grade_letter == "A"
        is_incorrect = grade_letter == "B"
        is_not_attempted = grade_letter == "C"
        
        return {
            "grade": grade_letter,
            "is_correct": is_correct,
            "is_incorrect": is_incorrect,
            "is_not_attempted": is_not_attempted,
        }

class SQA_CGAAggregator(NumericalAggregator):
    """This class implements a custom aggregator that computes accuracy as the ratio of correct to attempted answers.
    """

    def __init__(self, is_correct_column_name, is_incorrect_column_name, is_not_attempted_column_name, output_dir, group_by=None, **kwargs):
        """
        args:
            - is_correct_column_name (str): The name of the column containing the correct values.
            - is_incorrect_column_name (str): The name of the column containing the incorrect values.
            - is_not_attempted_column_name (str): The name of the column containing the not attempted values.
            - output_dir (str): The directory where the aggregated result will be stored.
            - groupby (str, optional): The column name to group the data by. Defaults to None.
        """
        super().__init__([is_correct_column_name, is_incorrect_column_name, is_not_attempted_column_name], output_dir, group_by=group_by, **kwargs)
        self.is_correct_column_name = is_correct_column_name
        self.is_incorrect_column_name = is_incorrect_column_name
        self.is_not_attempted_column_name = is_not_attempted_column_name

    def _aggregate(self, data):
        total_attempted = data[self.is_correct_column_name].sum() + data[self.is_incorrect_column_name].sum()
        if total_attempted == 0:
            divided_result = 0.0  # Avoid division by zero; define accuracy as 0 if no attempts were made
        else:
            divided_result = data[self.is_correct_column_name].sum() / total_attempted # TODO: handle NaNs, negative nums if any
        self.aggregated_result = {"accuracy_given_attempted": divided_result}

    def _aggregate_grouped(self, data):
        gb = data.groupby(self.group_by)
        total_attempted = gb[self.is_correct_column_name].sum() + gb[self.is_incorrect_column_name].sum()
        # Avoid division by zero; define accuracy as 0 if no attempts were made
        divided_result = (gb[self.is_correct_column_name].sum() / total_attempted.replace(0, 1)).to_dict() # TODO: handle NaNs, negative nums if any
        self.aggregated_result = {"accuracy_given_attempted": divided_result}


class SQA_CGAAvgPass1Aggregator(SQA_CGAAggregator):
    """This class implements a custom aggregator that computes accuracy as the ratio of correct to attempted answers.
    """

    def __init__(self, is_correct_column_name, is_incorrect_column_name, is_not_attempted_column_name, output_dir, group_by=None, **kwargs):
        """
        args:
            - is_correct_column_name (str): The name of the column containing the correct values.
            - is_incorrect_column_name (str): The name of the column containing the incorrect values.
            - is_not_attempted_column_name (str): The name of the column containing the not attempted values.
            - output_dir (str): The directory where the aggregated result will be stored.
            - groupby (str, optional): The column name to group the data by. Defaults to None.
        """
        super().__init__(is_correct_column_name, is_incorrect_column_name, is_not_attempted_column_name, output_dir, group_by=group_by, **kwargs)

    def _aggregate(self, data):
        if 'data_repeat_id' not in data.columns:
            super()._aggregate(data)
        else:
            self.group_by = 'data_repeat_id'
            super()._aggregate_grouped(data)            
            # Calculate the mean of the grouped results
            grouped_results = self.aggregated_result["accuracy_given_attempted"]
            mean_divided_result = sum(grouped_results.values()) / len(grouped_results) if len(grouped_results) > 0 else 0.0
            self.aggregated_result = {"accuracy_given_attempted": mean_divided_result}
    
    def _aggregate_grouped(self, data):
        if self.group_by == 'data_repeat_id':
            self._aggregate(data)
        else:
            original_group_by = self.group_by
            gb = data.groupby(original_group_by)
            # For each group, apply the _aggregate method
            group_results = {}
            for name, group in gb:
                self._aggregate(group)
                group_results[name] = self.aggregated_result["accuracy_given_attempted"]
            self.aggregated_result = {"accuracy_given_attempted": group_results}