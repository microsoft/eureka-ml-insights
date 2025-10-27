"""Defines an aggregator that counts unique error messages in LiveCodeBench test results."""

import pandas as pd

from collections import Counter

from eureka_ml_insights.metrics import reports


class CountUniqueErrorMessagesAggregator(reports.Aggregator):

    def __init__(self,
                 error_messages_column_name: str,
                 output_dir: str,
                 exclude_empty: bool = True,
                 group_by: str | None = None,
                 **kwargs):
        """Aggregator that counts unique error messages.

        Args:
            error_message_column_name: Name of the column containing error
                messages. If the column contains lists of error messages, all
                messages will be counted.
            output_dir: Directory to save the aggregation results.
            exclude_empty: Whether to exclude empty error messages from the
                count.
            group_by: Optional column name to group by before aggregation.
            **kwargs: Additional arguments for the base Aggregator class.
        """
        super().__init__([error_messages_column_name],
                         output_dir,
                         group_by=group_by,
                         **kwargs)
        self._error_messages_column_name = error_messages_column_name
        self._exclude_empty = exclude_empty

    def _aggregate(self, data: pd.DataFrame) -> None:
        error_messages = data[self._error_messages_column_name].dropna()

        self.aggregated_result = self._get_count(error_messages)

    def _aggregate_grouped(self, data: pd.DataFrame) -> None:
        grouped = data.groupby(self.group_by)

        frequency_by_group: dict[str, dict[str, int]] = {}

        for group_name, group_data in grouped:
            error_messages = (
                group_data[self._error_messages_column_name].dropna())

            frequency_by_group[str(group_name)] = self._get_count(
                error_messages)

        self.aggregated_result = frequency_by_group

    def _get_count(self, error_messages: pd.Series) -> dict[str, int]:
        """Counts unique error messages.

        Args:
            error_messages: Series containing error messages or lists of
                error messages.
        
        Returns:
            A dictionary mapping each unique error message to its count.
        """
        # Convert all entries to list if they are not already
        normalized_error_message: list[list[str]] = (error_messages.apply(
            lambda x: [x] if isinstance(x, str) else x).to_list())

        # Flatten the list of lists into a single list
        flattened_error_messages: list[str] = [
            message for sublist in normalized_error_message
            for message in sublist
            if not self._exclude_empty or message.strip()
        ]

        # Maps error message to its count
        return Counter(flattened_error_messages)
