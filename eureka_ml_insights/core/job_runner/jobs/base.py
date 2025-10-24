"""Defines the base Job protocol for job execution."""

from typing import Any, Protocol


class Job(Protocol):
    """Protocol for a job to be executed by the job executor."""

    def get_command(self) -> list[str]:
        """Returns the command to execute the job."""
        ...

    def serialize_input(self) -> bytes:
        """Serializes the input needed to execute the job."""
        ...

    def deserialize_result(self, stdout: bytes, stderr: bytes,
                           retcode: int) -> Any:
        """Deserializes the job result from the runner output.

        Args:
            stdout: The standard output from the job runner.
            stderr: The standard error from the job runner.
            retcode: The return code from the job runner.
        """
        ...
