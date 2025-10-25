"""Defines utilities to evaluate functional test cases."""

import datetime

from typing import cast

from eureka_ml_insights.core.job_runner import job_runner
from eureka_ml_insights.core.job_runner.command_runners import base as command_runners_base
from eureka_ml_insights.core.job_runner.jobs import python_function_job
from eureka_ml_insights.metrics.live_code_bench import functional_test_case
from eureka_ml_insights.metrics.live_code_bench import test_case_result


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
        if (job_result.success
                and job_result.return_value == test_case.expected_output):
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
