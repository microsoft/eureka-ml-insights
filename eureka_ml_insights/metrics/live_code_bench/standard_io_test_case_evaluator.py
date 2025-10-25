import datetime

from typing import cast

from eureka_ml_insights.core.job_runner import job_runner
from eureka_ml_insights.core.job_runner.command_runners import base as command_runners_base
from eureka_ml_insights.core.job_runner.jobs import python_script_job
from eureka_ml_insights.metrics.live_code_bench import standard_io_test_case
from eureka_ml_insights.metrics.live_code_bench import test_case_result


def evaluate_standard_io_test_case(
    src_code: str,
    test_case: standard_io_test_case.StandardIOTestCase,
    runner: command_runners_base.CommandRunner,
    timeout: datetime.timedelta | None = None,
) -> test_case_result.TestCaseResult:
    """Evaluates a standard I/O test case against the provided source code.

    The source code is expected to be a script that reads from stdin and
    writes the answer to stdout.

    Args:
        src_code: The source code to be tested.
        test_case: The StandardIOTestCase instance.
        runner: The command runner to use for executing the job.
        timeout: An optional timeout for the code execution.
    
    Returns:
        A TestCaseResult instance indicating whether the test case passed.
    """
    job = python_script_job.PythonScriptJob(
        script=src_code,
        stdin=test_case.stdin,
    )

    result: job_runner.JobExecutionResult = job_runner.run_job(
        job=job, command_runner=runner, timeout=timeout)

    if result.runner_status == command_runners_base.CommandStatus.COMPLETED:
        job_result = cast(python_script_job.PythonScriptJobResult,
                          result.job_result)
        received_stdout: str = job_result.stdout_str.rstrip("\n")
        expected_stdout: str = test_case.expected_stdout.rstrip("\n")

        if job_result.returncode == 0 and received_stdout == expected_stdout:
            return test_case_result.TestCaseResult(passed=True)
        elif job_result.returncode != 0:
            return test_case_result.TestCaseResult(
                passed=False,
                error_message=(
                    f"Script exited with return code {job_result.returncode}. "
                    f"Stderr: {job_result.stderr_str}"))
        else:
            return test_case_result.TestCaseResult(
                passed=False,
                error_message=(f"Expected stdout: {expected_stdout}, "
                               f"but got: {received_stdout}"))
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
