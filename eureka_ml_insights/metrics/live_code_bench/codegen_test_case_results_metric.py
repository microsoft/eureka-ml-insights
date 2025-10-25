"""Defines the CodegenTestCaseResultsMetric.

This metric evaluates generated code against provided test cases.

This attempts to reproduce the behavior of
https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/runner/custom_evaluator.py.
"""

import concurrent.futures
import datetime
import pandas as pd

from typing import TypedDict, cast
from tqdm.auto import tqdm

from eureka_ml_insights.metrics import metrics_base
from eureka_ml_insights.core.job_runner.command_runners import base as command_runners_base
from eureka_ml_insights.metrics import python_code_utils
from eureka_ml_insights.metrics.live_code_bench import raw_test_case
from eureka_ml_insights.metrics.live_code_bench import functional_test_case
from eureka_ml_insights.metrics.live_code_bench import standard_io_test_case
from eureka_ml_insights.metrics.live_code_bench import functional_test_case_parser
from eureka_ml_insights.metrics.live_code_bench import standard_io_test_case_parser
from eureka_ml_insights.metrics.live_code_bench import functional_test_case_evaluator
from eureka_ml_insights.metrics.live_code_bench import standard_io_test_case_evaluator
from eureka_ml_insights.metrics.live_code_bench import test_case_result


class TestResultsMetric(TypedDict):
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
    all_passed: bool


class CodegenTestCaseResultsMetric(metrics_base.CompositeMetric):
    """Evaluates the generated code against the provided test cases.

    See __init__ for details on the expected columns in the DataFrame.
    See __evaluate__ for details on the output format.
    """

    def __init__(self,
                 code_column_name: str,
                 test_cases_column_name: str,
                 metadata_column_name: str,
                 runner: command_runners_base.CommandRunner,
                 timeout: datetime.timedelta | None = None,
                 max_workers: int = 1,
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
            additional_imports: Additional Python import statements to include
                at the start of the code being tested.
        """
        self._code_column_name = code_column_name
        self._test_cases_column_name = test_cases_column_name
        self._metadata_column_name = metadata_column_name
        self._runner = runner
        self._timeout = timeout
        self._max_workers = max_workers
        self._additional_imports = additional_imports

    # Override this method to show that a progress bar.
    # Otherwise, the behavior is the same as the parent.
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

    def __evaluate__(self,   # type: ignore
                     row: "pd.Series") -> TestResultsMetric:
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

        raw_test_cases: list[dict[str,
                                  str]] = row[self._test_cases_column_name]

        if not raw_test_cases:
            return TestResultsMetric(
                passed=[],
                error_messages=[],
                all_passed=False,
            )

        if not code:
            return TestResultsMetric(
                passed=[False] * len(raw_test_cases),
                error_messages=["No code generated."] * len(raw_test_cases),
                all_passed=False,
            )

        # Prepend additional imports to the code
        if self._additional_imports:
            code = self._additional_imports + "\n\n" + code

        is_valid_code, parse_error = python_code_utils.is_python_code_valid(
            code)
        if not is_valid_code:
            return TestResultsMetric(
                passed=[False] * len(raw_test_cases),
                error_messages=[f"Code has syntax error: {parse_error}"] *
                len(raw_test_cases),
                all_passed=False,
            )

        function_path = ""
        function_parsing_error = ""
        try:
            function_path = (python_code_utils.find_function_path(
                code, function_name) if function_name else "")
        except Exception as e:
            function_parsing_error = str(e)

        results: TestResultsMetric = {
            "passed": [],
            "error_messages": [],
            "all_passed": False,
        }
        max_workers: int = min(self._max_workers, len(raw_test_cases))

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers) as executor:
            futures: list[concurrent.futures.Future] = []
            for raw_tc in raw_test_cases:
                futures.append(
                    executor.submit(_run_test, raw_tc, code, function_path,
                                    function_parsing_error, self._runner,
                                    self._timeout))

            # Iterate in the same order as the test cases
            for future in futures:
                test_result = future.result()
                results["passed"].append(test_result.passed)
                results["error_messages"].append(test_result.error_message)

            results["all_passed"] = all(results["passed"])

        return results


def _parse_test_case(
    raw_tc: dict[str, str],
) -> functional_test_case.FunctionalTestCase | standard_io_test_case.StandardIOTestCase:
    """Parses a raw test case dictionary into a test case object.

    Args:
        raw_tc: A dictionary representing the test case.
    
    Returns:
        A FunctionalTestCase or StandardIOTestCase object.
    """
    require_keys = {"testtype", "input", "output"}
    missing_keys = require_keys - raw_tc.keys()
    if missing_keys:
        raise ValueError(f"Test case is missing required keys: {missing_keys}")

    validated_raw_tc = cast(raw_test_case.RawTestCaseDict, raw_tc)
    if validated_raw_tc["testtype"] == "functional":
        return functional_test_case_parser.parse_functional_test_case(
            validated_raw_tc)
    elif validated_raw_tc["testtype"] == "stdin":
        return standard_io_test_case_parser.parse_standard_io_test_case(
            validated_raw_tc)
    else:
        raise ValueError(
            f"Unknown test case type: {validated_raw_tc['testtype']}")


def _evaluate_test_case(
    src_code: str,
    test_case: functional_test_case.FunctionalTestCase
    | standard_io_test_case.StandardIOTestCase,
    runner: command_runners_base.CommandRunner,
    function_name: str = "",
    function_parsing_error: str = "",
    timeout: datetime.timedelta | None = None,
) -> test_case_result.TestCaseResult:
    """Evaluates a test case against the provided source code.

    Args:
        src_code: The source code to be tested.
        test_case: The test case to evaluate.
        runner: The command runner to use for executing the job.
        function_name: The name of the function to be tested. This is only
            required for FunctionalTestCase instances. If the function is part
            of a class, provide the full path (e.g., 'MyClass.my_function').
        timeout: An optional timeout for the code execution.

    Returns:
        A TestCaseResult instance indicating whether the test case passed.
    
    Raises:
        ValueError: If the test case type is unknown or if function_name is not
            provided for FunctionalTestCase.
    """
    if isinstance(test_case, functional_test_case.FunctionalTestCase):
        if function_name and function_parsing_error:
            raise ValueError("Cannot evaluate functional test case because "
                             f"parsing of function {function_name} failed: "
                             f"{function_parsing_error}")

        if not function_name:
            raise ValueError(
                "function_name must be provided for FunctionalTestCase.")

        return functional_test_case_evaluator.evaluate_functional_test_case(
            src_code=src_code,
            function_name=function_name,
            test_case=test_case,
            runner=runner,
            timeout=timeout,
        )
    elif isinstance(test_case, standard_io_test_case.StandardIOTestCase):
        return standard_io_test_case_evaluator.evaluate_standard_io_test_case(
            src_code=src_code,
            test_case=test_case,
            runner=runner,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown test case type: {type(test_case)}")


def _run_test(
    raw_test_case: dict[str, str],
    code: str,
    function_path: str,
    function_parsing_error: str,
    runner: command_runners_base.CommandRunner,
    timeout: datetime.timedelta | None = None,
) -> test_case_result.TestCaseResult:
    """Runs a single test case against the generated code."""
    try:
        test_case = _parse_test_case(raw_test_case)
    except Exception as e:
        return test_case_result.TestCaseResult(
            passed=False, error_message=f"Failed to parse test case: {str(e)}")

    try:
        return _evaluate_test_case(
            src_code=code,
            function_name=function_path,
            function_parsing_error=function_parsing_error,
            test_case=test_case,
            runner=runner,
            timeout=timeout,
        )
    except Exception as e:
        return test_case_result.TestCaseResult(
            passed=False,
            error_message=f"Error during test case evaluation: {str(e)}")
