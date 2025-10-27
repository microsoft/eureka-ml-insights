"""Defines utilities to evaluate functional test cases."""

import datetime

from typing import Any, cast

from eureka_ml_insights.core.job_runner import job_runner
from eureka_ml_insights.core.job_runner.command_runners import base as command_runners_base
from eureka_ml_insights.core.job_runner.jobs import python_function_job
from eureka_ml_insights.metrics.live_code_bench import functional_test_case
from eureka_ml_insights.metrics.live_code_bench import test_case_result


def _normalize_output(output: Any) -> Any:
    """Normalizes the output for comparison.

    Recursively converts lists to tuples and normalizes dictionary keys
    and values.

    This is similar to the normalization done in
    https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/evaluation/testing_util.py#L266

    Args:
        output: The output to normalize.

    Returns:
        The normalized output.
    """
    if isinstance(output, list) or isinstance(output, tuple):
        return tuple(_normalize_output(item) for item in output)
    elif isinstance(output, dict):
        return {
            _normalize_output(k): _normalize_output(v)
            for k, v in output.items()}

    return output


def evaluate_functional_test_case(
    src_code: str,
    function_name: str,
    test_case: functional_test_case.FunctionalTestCase,
    runner: command_runners_base.CommandRunner,
    timeout: datetime.timedelta | None = None,
) -> test_case_result.TestCaseResult:
    """Evaluates a functional test case against the provided source code.

    Args:
        src_code: The source code containing the function definition.
        function_name: The name of the function to be tested. If the function
            is part of a class, provide the full path (e.g.,
            'MyClass.my_function').
        test_case: The FunctionalTestCase instance.
        runner: The command runner to use for executing the job.
        timeout: An optional timeout for the code execution.

    Returns:
        A TestCaseResult instance indicating whether the test case passed.

    Raises:
        ValueError: If function_name is not provided.
    """
    if not function_name:
        raise ValueError(
            "function_name must be provided for FunctionalTestCase.")

    job = python_function_job.PythonFunctionJob(
        src_script=src_code,
        function_name=function_name,
        args=test_case.inputs,
    )

    result: job_runner.JobExecutionResult = job_runner.run_job(
        job=job, command_runner=runner, timeout=timeout)

    if result.runner_status == command_runners_base.CommandStatus.COMPLETED:
        job_result = cast(
            python_function_job.PythonFunctionJobResult,
            result.job_result)

        normalized_actual_output = _normalize_output(job_result.return_value)
        normalized_expected_output = _normalize_output(
            test_case.expected_output)

        if (job_result.success
            and normalized_actual_output == normalized_expected_output):
            return test_case_result.TestCaseResult(passed=True)
        elif not job_result.success:
            assert job_result.exception_class_name is not None
            assert job_result.exception_msg is not None
            return test_case_result.TestCaseResult(
                passed=False,
                error_message=("Raised exception " +
                               job_result.exception_class_name + ": " +
                               job_result.exception_msg))
        else:
            return test_case_result.TestCaseResult(
                passed=False,
                error_message=(
                    f"Expected output: {test_case.expected_output}, "
                    f"but got: {job_result.return_value}"))
    elif result.runner_status == command_runners_base.CommandStatus.TIMEOUT:
        assert timeout is not None
        return test_case_result.TestCaseResult(
            passed=False,
            error_message=("Code execution timed out after "
                           f"{timeout.total_seconds()} seconds."))
    elif result.runner_status == command_runners_base.CommandStatus.FAILED_TO_RUN:
        return test_case_result.TestCaseResult(
            passed=False,
            error_message=(
                f"Code execution failed to run: {result.job_result.stderr}"))
    else:
        raise ValueError(f"Unknown runner status: {result.runner_status}")
