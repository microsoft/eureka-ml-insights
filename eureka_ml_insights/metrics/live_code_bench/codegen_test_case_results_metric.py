"""Defines the CodegenTestCaseResultsMetric.

This metric evaluates generated code against provided test cases.
"""

import datetime
import pandas as pd

from collections import defaultdict

from eureka_ml_insights.metrics import metrics_base
from eureka_ml_insights.metrics.live_code_bench import (
    evaluate_codegen,
    code_parsing,
)


class CodegenTestCaseResultsMetric(metrics_base.CompositeMetric):
    """Evaluates the generated code against the provided test cases.

    See __init__ for details on the expected columns in the DataFrame.
    See __evaluate__ for details on the output format.
    """

    def __init__(self, code_column_name: str, public_test_cases_column_name: str,
                 private_test_cases_column_name: str,
                 metadata_column_name: str,
                 timeout: datetime.timedelta | None = None):
        """Initializes the CodegenTestCaseResultsMetric.

        Args:
            code_column_name: The name of the column containing the code as a
                string.
            public_test_cases_column_name: The name of the column containing the
                list of public test case dictionaries. The column should contain
                a string representation of a list of dictionaries, each
                representing a test case. See __evaluate__ method for details.
            private_test_cases_column_name: The name of the column containing the
                list of private test case dictionaries. The column should contain
                a string representation of a list of dictionaries, each
                representing a test case. See __evaluate__ method for details.
            metadata_column_name: The name of the column containing metadata
                dictionary. The column should contain a string representation of
                a dictionary. See __evaluate__ method for details.
            timeout: An optional timeout for each test case execution.
        """
        self._code_column_name = code_column_name
        self._public_test_cases_column_name = public_test_cases_column_name
        self._private_test_cases_column_name = private_test_cases_column_name
        self._metadata_column_name = metadata_column_name
        self._timeout = timeout

    def __evaluate__(self, row: "pd.Series") -> dict[str, list[bool | str]]:  # type: ignore
        """Runs the code against the test cases and checks if they pass.

        Args:
            row: A row from the DataFrame containing the code and test cases.
                Must have the following columns:
                - code_column_name: The generated code as a string.
                - (public_test_cases_column_name and
                    private_test_cases_column_name): A list of test case
                    dictionaries.
                    Each test case dictionary should have the following keys:
                        - 'testtype': Either 'functional' or 'stdin'.
                        - 'input': The input for the test case.
                        - 'output': The expected output for the test case.
                - metadata_column_name: A dictionary with metadata, including
                    the function name under the key 'func_name'. This is only
                    needed for functional test cases.

        Returns:
            A dictionary with keys:
            - 'passed': A list of booleans indicating if each test case passed.
            - 'error_message': A list of strings with error messages for each
                failed test case. If no error, the string is empty.
            The lists are in the same order as the test cases.
        """
        code: str = row[self._code_column_name]
        function_name: str = row[self._metadata_column_name].get(
            "func_name", "")

        public_test_cases: list[dict[str, str]] = (
            row[self._public_test_cases_column_name])
        private_test_cases: list[dict[str, str]] = (
            row[self._private_test_cases_column_name])

        raw_test_cases: list[dict[str, str]] = (
            public_test_cases + private_test_cases)

        if not raw_test_cases:
            return {"passed": [], "error_messages": []}

        is_valid_code, parse_error = code_parsing.is_python_code_valid(code)
        if not is_valid_code:
            return {
                "passed": [False] * len(raw_test_cases),
                "error_messages": [
                    f"Code has syntax error: {parse_error}"
                ] * len(raw_test_cases),
            }
        
        function_path = ""
        function_parsing_error = ""
        try:
            function_path = (
                code_parsing.find_function_path(code, function_name)
                if function_name else ""
            )
        except Exception as e:
            function_parsing_error = str(e)

        results: dict[str, list[bool | str]] = defaultdict(list)
        for raw_test_case in raw_test_cases:
            try:
                test_case = evaluate_codegen.parse_test_case(raw_test_case)
                if (isinstance(test_case, evaluate_codegen.FunctionalTestCase)
                        and function_parsing_error):
                    test_result = evaluate_codegen.TestCaseResult(
                        passed=False,
                        error_message=(
                            f"Cannot run functional test case because "
                            f"function parsing failed: {function_parsing_error}"
                        ),
                    )
                    continue
                test_result = evaluate_codegen.evaluate_test_case(
                    src_code=code,
                    function_name=function_path,
                    test_case=test_case,
                    timeout=self._timeout,
                )
            except Exception as e:
                test_result = evaluate_codegen.TestCaseResult(
                    passed=False,
                    error_message=f"Unexpected error: {str(e)}"
                )

            results["passed"].append(test_result.passed)
            results["error_messages"].append(test_result.error_message)

        return results