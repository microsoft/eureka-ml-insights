"""Defines the interface for executing CLI commands."""

import dataclasses
import datetime

from enum import Enum
from typing import Protocol


class CommandStatus(Enum):
    """Status of a command execution."""
    COMPLETED = "completed"
    FAILED_TO_RUN = "failed_to_run"
    TIMEOUT = "timeout"


@dataclasses.dataclass(frozen=True)
class CommandResult:
    """Result of running a command.

    Attributes:
        stdout: The standard output from the command execution.
        stderr: The standard error from the command execution.
        returncode: The return code from the command execution.
        status: The status of the command execution.
    """
    stdout: bytes = b""
    stderr: bytes = b""
    returncode: int = 0
    status: CommandStatus = CommandStatus.COMPLETED


class CommandRunner(Protocol):
    """Protocol for running a command and returning the result."""

    def run(
        self,
        command: list[str],
        stdin: bytes | None = None,
        timeout: datetime.timedelta | None = None,
    ) -> CommandResult:
        """
        Executes a command and returns the result.

        Args:
            command: List of command-line arguments,
                e.g., ['python', '-c', 'print(1)'].
            stdin: Optional bytes to provide as stdin to the process.
            timeout: Timeout for the command execution. If the command
                exceeds this duration, it should be terminated.

        Returns:
            The result of the command execution.
        """
        ...
