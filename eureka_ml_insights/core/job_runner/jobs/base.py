"""Defines the base Job protocol for job execution.

The job knows how to serialize its input and the command that it needs to run.
It also knows how to interpret (deserialize) the output produced by executing
the job.
"""

import abc

from typing import Any


class Job(abc.ABC):
    """Protocol for a job to be executed by the job executor."""

    @abc.abstractmethod
    def get_command(self) -> list[str]:
        """Returns the command to execute the job."""

    @abc.abstractmethod
    def serialize_input(self) -> bytes:
        """Serializes the input needed to execute the job."""

    @abc.abstractmethod
    def deserialize_result(self, stdout: bytes, stderr: bytes,
                           retcode: int) -> Any:
        """Deserializes the job result from the runner output.

        Args:
            stdout: The standard output from the job runner.
            stderr: The standard error from the job runner.
            retcode: The return code from the job runner.
        """
