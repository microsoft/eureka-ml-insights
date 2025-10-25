"""Defines the job runner that executes jobs using a specified command runner."""

import datetime
import dataclasses

from typing import Any

from eureka_ml_insights.core.job_runner.jobs import base as jobs_base
from eureka_ml_insights.core.job_runner.command_runners import base as command_runners_base


@dataclasses.dataclass(frozen=True)
class JobExecutionResult:
    """Result of a job execution.

    Attributes:
        job_result: The result of the job execution.
        runner_status: The status of the command runner after execution.
    """
    job_result: Any
    runner_status: command_runners_base.CommandStatus


def run_job(job: jobs_base.Job,
            command_runner: command_runners_base.CommandRunner,
            timeout: datetime.timedelta | None = None) -> JobExecutionResult:
    """Runs a job using the specified command runner.

    Args:
        job: The job to execute.
        command_runner: The command runner to use for execution.
        stdin: Optional standard input to provide to the job.
        timeout: Optional timeout for the job execution.

    Returns:
        The result of the job execution.
    """
    command = job.get_command()
    serialized_input = job.serialize_input()

    command_result = command_runner.run(command=command,
                                        stdin=serialized_input,
                                        timeout=timeout)

    job_result = job.deserialize_result(stdout=command_result.stdout,
                                        stderr=command_result.stderr,
                                        retcode=command_result.returncode)

    return JobExecutionResult(job_result=job_result,
                              runner_status=command_result.status)
