"""Defines the CodegenTestCaseResultsMetric.

This metric evaluates generated code against provided test cases.
"""

import concurrent.futures
import datetime
import pandas as pd

from typing import TypedDict, override
from tqdm.auto import tqdm

from eureka_ml_insights.metrics import metrics_base
from eureka_ml_insights.metrics.live_code_bench import (
    evaluate_codegen,
    code_parsing,
    sandbox_config,
)


class TestResults(TypedDict):
    """Defines the structure of the test results returned by the metric.

    Attributes:
        passed: A list of booleans indicating whether each test case passed.
            In the same order as the test cases provided.
        error_messages: A list of error messages for each test case that failed.
            In the same order as the test cases provided. If a test case passed,
            the corresponding error message is an empty string.
        all_passed: A boolean indicating whether all test cases passed.
            None only if there are no test cases.
    """

    passed: list[bool]
    error_messages: list[str]
    all_passed: bool | None


def _run_test(
        raw_test_case: dict[str, str],
        code: str,
        function_path: str,
        function_parsing_error: str,
        timeout: datetime.timedelta | None = None,
        sandbox_cfg: sandbox_config.SandboxConfig | None = None,
) -> evaluate_codegen.TestCaseResult:
    """Runs a single test case against the generated code.

    Args:
        raw_test_case: A dictionary representing the test case.
            Should have the following keys:
                - 'testtype': Either 'functional' or 'stdin'.
                - 'input': The input for the test case.
                - 'output': The expected output for the test case.
        code: The generated code as a string.
        function_path: The path to the function to be tested.
            Empty string if not applicable.
        function_parsing_error: An error message if there was an error
            parsing the function, empty string otherwise.
        timeout: An optional timeout for the test case execution.
        sandbox_cfg: An optional SandboxConfig to run the code in a sandbox.

    Returns:
        The result of the test.
    """
    try:
        test_case = evaluate_codegen.parse_test_case(raw_test_case)
        if (isinstance(test_case, evaluate_codegen.FunctionalTestCase)
            and function_parsing_error):
            return evaluate_codegen.TestCaseResult(
                passed=False,
                error_message=(
                    f"Cannot run functional test case because "
                    f"function parsing failed: {function_parsing_error}"
                ),
            )
        return evaluate_codegen.evaluate_test_case(
            src_code=code,
            function_name=function_path,
            test_case=test_case,
            timeout=timeout,
            sandbox_cfg=sandbox_cfg,
        )
    except Exception as e:
        return evaluate_codegen.TestCaseResult(
            passed=False,
            error_message=f"Unexpected error: {str(e)}"
        )


class CodegenTestCaseResultsMetric(metrics_base.CompositeMetric):
    """Evaluates the generated code against the provided test cases.

    See __init__ for details on the expected columns in the DataFrame.
    See __evaluate__ for details on the output format.
    """

    def __init__(self, code_column_name: str, test_cases_column_name: str,
                 metadata_column_name: str,
                 timeout: datetime.timedelta | None = None,
                 max_workers: int = 1,
                 sandbox_cfg: sandbox_config.SandboxConfig | None = None,
                 additional_imports: str = "") -> None:
        """Initializes the CodegenTestCaseResultsMetric.

        Args:
            code_column_name: The name of the column containing the code as a
                string.
            test_cases_column_name: The name of the column containing the
                list of test case dictionaries. The column should contain
                a string representation of a list of dictionaries, each
                representing a test case. See __evaluate__ method for details.
            metadata_column_name: The name of the column containing metadata
                dictionary. The column should contain a string representation of
                a dictionary. See __evaluate__ method for details.
            timeout: An optional timeout for each test case execution.
            max_workers: The maximum number of workers to use for parallel
                execution of test cases.
            sandbox_cfg: An optional SandboxConfig to run the code in a sandbox.
            additional_imports: Additional Python import statements to include
                at the start of the code being tested.
        """
        self._code_column_name = code_column_name
        self._test_cases_column_name = test_cases_column_name
        self._metadata_column_name = metadata_column_name
        self._timeout = timeout
        self._max_workers = max_workers
        self._sandbox_cfg = sandbox_cfg
        self._additional_imports = additional_imports

    # Override this method to show that a progress bar.
    # Otherwise, the behavior is the same as the parent.
    @override
    def evaluate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Evaluates the generated code against the provided test cases.

        Args:
            data: A DataFrame containing the data to evaluate. See
                __evaluate__ method for details on the expected columns.

        Returns:
            A DataFrame with an additional columns.
        """
        self.validate_data(data)

        tqdm.pandas(desc="Running test cases against generated code")

        data[self.__class__.__name__ + "_result"] = (
            data.progress_apply(lambda x: self.__evaluate__(x), axis=1))  # type: ignore

        data = self.decompose_metric(data)

        return data

    @override
    def __evaluate__(self, row: "pd.Series") -> TestResults:  # type: ignore
        """Runs the code against the test cases and checks if they pass.

        Args:
            row: A row from the DataFrame containing the code and test cases.
                Must have the following columns:
                - code_column_name: The generated code as a string.
                - test_cases_column_name: A list of test case dictionaries.
                    Each test case dictionary should have the following keys:
                        - 'testtype': Either 'functional' or 'stdin'.
                        - 'input': The input for the test case.
                        - 'output': The expected output for the test case.
                - metadata_column_name: A dictionary with metadata, including
                    the function name under the key 'func_name'. This is only
                    needed for functional test cases.

        Returns:
            A TestResults dictionary.
        """
        code: str = row[self._code_column_name].strip()
        function_name: str = row[self._metadata_column_name].get(
            "func_name", "")

        raw_test_cases: list[dict[str, str]] = row[self._test_cases_column_name]

        if not raw_test_cases:
            return TestResults(
                passed=[],
                error_messages=[],
                all_passed=None,
            )

        if not code:
            return TestResults(
                passed=[False] * len(raw_test_cases),
                error_messages=[
                    "No code generated."
                ] * len(raw_test_cases),
                all_passed=False,
            )

        # Prepend additional imports to the code
        if self._additional_imports:
            code = self._additional_imports + "\n\n" + code

        is_valid_code, parse_error = code_parsing.is_python_code_valid(code)
        if not is_valid_code:
            return TestResults(
                passed=[False] * len(raw_test_cases),
                error_messages=[
                    f"Code has syntax error: {parse_error}"
                ] * len(raw_test_cases),
                all_passed=False,
            )
        
        function_path = ""
        function_parsing_error = ""
        try:
            function_path = (
                code_parsing.find_function_path(code, function_name)
                if function_name else ""
            )
        except Exception as e:
            function_parsing_error = str(e)

        results: TestResults = {
            "passed": [],
            "error_messages": [],
            "all_passed": False,
        }
        max_workers: int = min(self._max_workers, len(raw_test_cases))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers) as executor:
            futures = [
                executor.submit(_run_test, raw, code, function_path,
                                function_parsing_error, self._timeout,
                                self._sandbox_cfg)
                for raw in raw_test_cases
            ]

            # Iterate in the same order as the test cases
            for future in futures:
                test_result = future.result()
                results["passed"].append(test_result.passed)
                results["error_messages"].append(test_result.error_message)

            results["all_passed"] = all(results["passed"])

        return results
