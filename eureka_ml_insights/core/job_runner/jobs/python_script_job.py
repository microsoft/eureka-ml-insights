"""Defines how to execute a Python script provided as source code."""

import dataclasses
import sys


@dataclasses.dataclass(frozen=True)
class PythonScriptFromSrcJobResult:
    """Result of executing a PythonScriptFromSrcJob.

    Attributes:
        stdout: The standard output from the script execution.
        stderr: The standard error from the script execution.
        returncode: The return code from the script execution.
        stdout_str: The standard output decoded as a UTF-8 string.
        stderr_str: The standard error decoded as a UTF-8 string.
    """
    stdout: bytes
    stderr: bytes
    returncode: int

    @property
    def stdout_str(self) -> str:
        return self.stdout.decode("utf-8", errors="replace")

    @property
    def stderr_str(self) -> str:
        return self.stderr.decode("utf-8", errors="replace")


@dataclasses.dataclass(frozen=True)
class PythonScriptFromSrcJob:
    """Job to execute a Python script provided as source code.

    The script is expected to read input from stdin and write output to stdout.
    It should be self-contained, including all necessary imports and
    definitions.

    Attributes:
        script: The Python script to execute as a string.
        stdin: The standard input to provide to the script.
    """
    script: str
    stdin: bytes | str = b""

    def get_command(self) -> list[str]:
        """Returns the command to execute the job in the command line."""
        return [sys.executable, "-c", self.script]

    def serialize_input(self) -> bytes:
        """Serializes the input needed to execute the job."""
        if isinstance(self.stdin, str):
            return self.stdin.encode("utf-8")
        return self.stdin

    def deserialize_result(
        self,
        stdout: bytes,
        stderr: bytes,
        retcode: int
    ) -> PythonScriptFromSrcJobResult:
        """Deserializes the job result from the runner output.

        Args:
            stdout: The standard output from the job runner.
            stderr: The standard error from the job runner.
            retcode: The return code from the job runner.

        Returns:
            The deserialized PythonScriptJobResult.
        """
        return PythonScriptFromSrcJobResult(
            stdout=stdout,
            stderr=stderr,
            returncode=retcode)
