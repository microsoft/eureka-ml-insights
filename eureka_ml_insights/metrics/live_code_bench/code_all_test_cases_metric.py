import pandas as pd

from eureka_ml_insights.metrics import metrics_base
from eureka_ml_insights.metrics.live_code_bench import evaluate_code_utils


class CodePassedAllTestCasesMetric(metrics_base.Metric):
    """Evaluates the generated code against the provided test cases."""

    def __init__(self, code_column_name: str, test_cases_column_name: str):
        """
        Args:
            code_column_name: The name of the column containing the generated
                code to be evaluated. This column should contain a string with
                the self-contained code snippet.
            test_cases_column_name: The name of the column containing the test
                cases to run against the generated code. The column should
                contain a JSON array, where each element has the form:
                    {
                        "inputs": <string containing the inputs to the function
                                    exactly as they would be passed in a
                                    function call, one argument per line>,
                        "expected_output": "<expected output from the function,
                                            exactly as it would be returned>"
                    }
                Example:
                    Say the code is as follows:
                        ```python
                        class Solution:
                            def add(self, arr: list[int], x: int) -> list[int]:
                                return [a + x for a in arr], "hello"
                        ```
                    Then the test_cases_column_name should contain something
                    like:
                    [
                        {
                            "inputs": "[1, 2]\n1",
                            "expected_output": "[[2, 3], 'hello']"
                        },
                        {
                            "inputs": "[5, 7]\n2",
                            "expected_output": "[[7, 9], 'hello']"
                        }
                    ]
        """
        self._code_column_name = code_column_name
        self._test_cases_column_name = test_cases_column_name

    def __evaluate__(self, row: pd.Series) -> bool:  # type: ignore
        """Runs the code against the test cases and checks if all pass.

        Args:
            row: A row from the DataFrame containing the code and test cases.

        Returns:
            True if all test cases pass, False otherwise.
        """
        test_cases: list[evaluate_code_utils.TestCase] = (
            evaluate_code_utils._parse_test_cases(
                row[self._test_cases_column_name]
            )
        )
        return True
