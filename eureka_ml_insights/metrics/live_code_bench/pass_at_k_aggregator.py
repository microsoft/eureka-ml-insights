"""Defines Pass@K metric aggregator."""

import pandas as pd

from typing import cast

from eureka_ml_insights.metrics import reports
from eureka_ml_insights.metrics.live_code_bench import pass_at_k


class PassAtKAggregator(reports.NumericalAggregator):
    """Implements the Pass@K metric aggregator."""

    def __init__(self, passed_column_name: str, k: int, **kwargs):
        """Initializes the Pass@K aggregator.

        Args:
            passed_column_name: The name of the column indicating whether an
                attempt passed all test cases. Each entry should be a boolean
                value. True indicates a passing attempt.
            k: The K value for the Pass@K metric.
            group_by: The name of the column to group by.
        """
        super().__init__(column_names=[passed_column_name], **kwargs)

        if k <= 0:
            raise ValueError(f"K must be a positive integer. Got {k}.")

        self._passed_column_name = passed_column_name
        self._k = k

    def _aggregate(self, data: pd.DataFrame) -> None:
        """Computes the Pass@K metric.

        Args:
            data: A pandas DataFrame containing the data to aggregate.

        Returns:
            None. The result is stored in self.aggregated_result, which is a
            dictionary with a single key-value pair where the key is
            "pass@K" (with K replaced by the actual K value) and the value is
            the computed Pass@K metric.
        """
        passed: pd.Series[bool] = data[self._passed_column_name]

        num_attempts: int = len(passed)
        num_correct: int = passed.sum()

        pass_at_k_estimate: float = pass_at_k.estimate_pass_at_k(
            num_attempts=num_attempts, num_correct=num_correct, k=self._k)

        self.aggregated_result = {f"pass@{self._k}": pass_at_k_estimate}

    def _aggregate_grouped(self, data: pd.DataFrame):
        """Computes the Pass@K metric for each group in the data.

        Args:
            data: A pandas DataFrame containing the data to aggregate.

        Returns:
            None. The result is stored in self.aggregated_result, which is a
            dictionary mapping each group to its computed Pass@K metric.
        """
        grouped_data = data.groupby(self.group_by)

        num_attempts_per_group: pd.Series[int] = grouped_data.size()
        num_correct_per_group: pd.Series[int] = (
            grouped_data[self._passed_column_name].sum())

        # Maps the group name to its Pass@K estimate
        pass_at_k_from_group: dict[str, float] = {}
        for group_name in grouped_data.groups.keys():
            group_name = cast(str, group_name)  # tell type checker it's a str
            num_attempts = int(num_attempts_per_group[group_name])
            num_correct = int(num_correct_per_group[group_name])

            pass_at_k_estimate = pass_at_k.estimate_pass_at_k(
                num_attempts=num_attempts,
                num_correct=num_correct,
                k=self._k,
            )

            pass_at_k_from_group[group_name] = pass_at_k_estimate

        self.aggregated_result = {f"pass@{self._k}": pass_at_k_from_group}
