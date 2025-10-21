"""Defines utilities for executing Python code and retrieving results.

Executing functions typical usage:
    job = FunctionJob(
        src_code="def add(a, b): return a + b",
        function_name="add",
        args=(2, 3),
        timeout=datetime.timedelta(seconds=5)
    )
    result: FunctionResult = execute_function(job)

Executing scripts typical usage:
    job = ScriptJob(
        script="print('Hello, World!')",
        timeout=datetime.timedelta(seconds=5)
    )
    result: ScriptResult = execute_script(job)
"""

import dataclasses
import datetime
import pickle
import subprocess
import sys

from enum import Enum
from typing import Any, Protocol

# Minimal wrapper script to run functions in a subprocess
# This script reads a pickled FunctionJob from stdin, executes the function,
# and writes the pickled result to stdout.
_FUNCTION_RUNNER_SCRIPT = """
import inspect
import io
import pickle
import sys
import traceback

from contextlib import redirect_stdout, redirect_stderr
from typing import Any
from collections.abc import Callable


def extract_function(
    function_name: str,
    namespace: dict[str, Any]
) -> Callable[..., Any]:
    '''
    Extracts a function or method from the given namespace.

    Args:
        function_name: Dotted path to the function or method (e.g.,
            'MyClass.my_method' or 'my_function').
        namespace: The namespace dictionary where the function/class is defined.
    
    Returns:
        The callable function or method.
    
    Raises:
        AttributeError: If any part of the dotted path is not found.
        TypeError: If the resolved object is not callable.
        KeyError: If a part of the path is not found in a dictionary.
    '''
    parts = function_name.split('.')
    obj = namespace
    
    # Traverse the dotted path
    for i, part in enumerate(parts):
        if isinstance(obj, dict):
            if part not in obj:
                raise KeyError(f"'{part}' not found in namespace")
            obj = obj[part]
        else:
            if not hasattr(obj, part):
                raise AttributeError(
                    f"'{type(obj).__name__}' object has no attribute '{part}'")
            obj = getattr(obj, part)

        # If we have more parts to traverse and current obj is a class,
        # instantiate it
        if i < len(parts) - 1 and inspect.isclass(obj):
            obj = obj()
    
    # Verify the final object is callable
    if not callable(obj):
        raise TypeError(
            f"'{function_name}' is not callable (got {type(obj).__name__})")
    
    return obj


# Read serialized input from stdin
input_data = pickle.load(sys.stdin.buffer)

src_code = input_data["src_code"]
function_name = input_data["function_name"]
args = input_data["args"]
kwargs = input_data["kwargs"]

code_object = compile(src_code, '<my_source>', 'exec')

# Prepare to capture output
# This is important to avoid mixing function output with the pickled result
# written to stdout.
stdout_buffer = io.StringIO()
stderr_buffer = io.StringIO()

try:
    namespace: dict[str, Any] = {}
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        exec(code_object, namespace)
        func = extract_function(function_name, namespace)
        result = func(*args, **kwargs)
    output = {
        "success": True,
        "result": result,
        "error": "",
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr_buffer.getvalue(),
    }
except Exception:
    output = {
        "success": False,
        "result": None,
        "error": traceback.format_exc(),
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr_buffer.getvalue(),
    }

# Write *only the pickle* to stdout.buffer
pickle.dump(output, sys.stdout.buffer)
"""


@dataclasses.dataclass(frozen=True)
class FunctionJob:
    """Represents a function to execute with bound arguments.

    Example of how to create a FunctionJob:
        job = FunctionJob(
            src_code="def add(a, b): return a + b",
            function_name="add",
            args=(2, 3)
        )
    
    Attributes:
        src_code: The source code of the script containing the function. The
            script should be self-contained (i.e., include all necessary
            imports).
        function_name: Name of the function to extract and execute.
        args: Positional arguments to pass to the function.
        kwargs: Keyword arguments to pass to the function.
    """
    src_code: str
    function_name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    timeout: datetime.timedelta | None = None


@dataclasses.dataclass(frozen=True)
class ScriptJob:
    """Represents a Python script to execute.
    
    Attributes:
        script: The self-contained Python code to execute.
        stdin_input: Optional byte sequence or string to provide as stdin to the
            script.
    """
    script: str
    stdin_input: str | bytes = b""
    timeout: datetime.timedelta | None = None


class ProcessExitReason(Enum):
    """Reasons for exit.

    Attributes:
        COMPLETED: Process completed normally.
        TIMEOUT: Process was terminated due to timeout.
        STARTUP_FAILURE: Process failed to start.
    """
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    STARTUP_FAILURE = "startup_failure"


@dataclasses.dataclass(frozen=True)
class RawProcessResult:
    """Raw result from executing a process.
    
    Attributes:
        stdout: Content written to standard output.
        stderr: Content written to standard error.
        returncode: Process return code.
        exit_reason: Reason for process exit.
        error_message: Error message if execution failed.
    """
    stdout: bytes = b""
    stderr: bytes = b""
    returncode: int = 0
    exit_reason: ProcessExitReason = ProcessExitReason.COMPLETED
    error_message: str = ""


@dataclasses.dataclass(frozen=True)
class ScriptResult:
    """Parsed result of executing a script.
    
    Attributes:
        stdout: Content written by the script to stdout.
        stderr: Content written by the script to stderr.
        error_message: Error message if execution failed, empty string
            otherwise.
        success: True iff execution completed without errors.
    """
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""

    @property
    def success(self) -> bool:
        """Returns True if execution completed without errors."""
        return not self.error_message


@dataclasses.dataclass(frozen=True)
class FunctionResult:
    """Result of executing a function.
    
    Attributes:
        return_value: The return value from the function.
        stdout: Content written by the function to stdout during execution.
        stderr: Content written by the function to stderr during execution.
        error_message: Error message if execution failed.
    """
    return_value: Any = None
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""

    @property
    def success(self) -> bool:
        """Returns True if execution completed without errors."""
        return not self.error_message


class ProcessRunner(Protocol):
    """Protocol for running subprocesses (enables testing with mocks)."""

    def run(self, args: list[str], input: bytes | None, capture_output: bool,
            timeout: datetime.timedelta | None) -> subprocess.CompletedProcess:
        """Run a subprocess and return the result."""
        ...


class SubprocessRunner:
    """Default subprocess runner using subprocess.run."""

    def run(self, args: list[str], input: bytes | None, capture_output: bool,
            timeout: datetime.timedelta | None) -> subprocess.CompletedProcess:
        return subprocess.run(
            args,
            input=input,
            capture_output=capture_output,
            timeout=timeout.total_seconds() if timeout else None)


def _execute_subprocess(
    args: list[str],
    runner: ProcessRunner,
    input: bytes = b"",
    timeout: datetime.timedelta | None = None,
) -> RawProcessResult:
    """Execute a subprocess with given arguments and input.
    
    Args:
        args: Command line arguments for the subprocess.
        runner: ProcessRunner for subprocess execution.
        input: Optional byte sequence to provide as stdin.
        timeout: Maximum allowed time for execution.
    
    Returns:
        RawProcessResult with captured outputs.
    """

    try:
        result = runner.run(args,
                            input=input,
                            capture_output=True,
                            timeout=timeout)

        return RawProcessResult(stdout=result.stdout,
                                stderr=result.stderr,
                                returncode=result.returncode)

    except subprocess.TimeoutExpired as e:
        return RawProcessResult(exit_reason=ProcessExitReason.TIMEOUT,
                                error_message=str(e))
    except Exception as e:
        return RawProcessResult(exit_reason=ProcessExitReason.STARTUP_FAILURE,
                                error_message=str(e))


def execute_script(job: ScriptJob,
                   runner: ProcessRunner | None = None) -> ScriptResult:
    """Execute a ScriptJob and retrieves the result.
    
    Args:
        job: The ScriptJob to execute.
        runner: ProcessRunner for execution.
    
    Returns:
        ScriptResult with captured outputs.
    
    Raises:
        ValueError: If an unknown exit reason is encountered.
    """
    if runner is None:
        runner = SubprocessRunner()

    input: bytes = (job.stdin_input.encode('utf-8') if isinstance(
        job.stdin_input, str) else job.stdin_input)

    raw_result = _execute_subprocess(args=[sys.executable, '-c', job.script],
                                     input=input,
                                     runner=runner,
                                     timeout=job.timeout)

    if raw_result.exit_reason == ProcessExitReason.COMPLETED:
        stderr_str = raw_result.stderr.decode("utf-8", errors="replace")

        if raw_result.returncode != 0:
            error_message = f"Script exited with code {raw_result.returncode}"
            if stderr_str.strip():
                error_message += f". Stderr: {stderr_str.strip()}"
        else:
            error_message = ""

        stdout_str = raw_result.stdout.decode("utf-8", errors="replace")

        return ScriptResult(stdout=stdout_str,
                            stderr=stderr_str,
                            error_message=error_message)
    elif raw_result.exit_reason == ProcessExitReason.TIMEOUT:
        timeout_seconds = (job.timeout.total_seconds()
                           if job.timeout else "unknown")
        return ScriptResult(
            error_message=
            f"Timeout: Script execution exceeded {timeout_seconds} "
            f"seconds.\n{raw_result.error_message}"
        )
    elif raw_result.exit_reason == ProcessExitReason.STARTUP_FAILURE:
        return ScriptResult(
            error_message=f"Startup failure: {raw_result.error_message}")
    else:
        raise ValueError(f"Unknown exit reason: {raw_result.exit_reason}")


def _serialize_function_job(job: FunctionJob) -> bytes:
    """Serialize a FunctionJob to bytes for subprocess communication.
    
    Args:
        job: The FunctionJob to serialize.
    
    Returns:
        Pickled bytes representing the job.
    """
    return pickle.dumps({
        "src_code": job.src_code,
        "function_name": job.function_name,
        "args": job.args,
        "kwargs": job.kwargs
    })


def _deserialize_function_result(
        raw_result: RawProcessResult) -> FunctionResult:
    """Deserialize the result from a function execution subprocess.

    Args:
        raw_result: The raw outputs from the subprocess.

    Returns:
        FunctionResult with captured outputs and errors.
    """
    stderr_str = raw_result.stderr.decode("utf-8", errors="replace")

    if raw_result.returncode != 0:
        return FunctionResult(
            error_message=f"Process exited with code {raw_result.returncode}. "
            f"Stderr: {stderr_str}")

    try:
        output = pickle.loads(raw_result.stdout)

        success = output["success"]
        result = output["result"]
        error = output["error"]
        captured_stdout = output["stdout"]
        captured_stderr = output["stderr"]

        if success:
            return FunctionResult(return_value=result,
                                  stdout=captured_stdout,
                                  stderr=captured_stderr)
        else:
            return FunctionResult(
                stdout=captured_stdout,
                stderr=captured_stderr,
                error_message=f"Exception during execution:\n{error}")

    except (pickle.UnpicklingError, KeyError, EOFError, TypeError) as e:
        return FunctionResult(
            stdout=raw_result.stdout.decode("utf-8", errors="replace"),
            stderr=stderr_str,
            error_message=f"Failed to deserialize result: {str(e)}")


def execute_function(job: FunctionJob,
                     runner: ProcessRunner | None = None) -> FunctionResult:
    """Execute a FunctionJob and retrieves the result.
    
    Args:
        job: The FunctionJob to execute.
        runner: ProcessRunner for subprocess execution.
    
    Returns:
        JobResult with captured outputs.
    
    Raises:
        ValueError: If an unknown exit reason is encountered.
    """
    if runner is None:
        runner = SubprocessRunner()

    raw_result = _execute_subprocess(
        args=[sys.executable, '-c', _FUNCTION_RUNNER_SCRIPT],
        input=_serialize_function_job(job),
        runner=runner,
        timeout=job.timeout)

    if raw_result.exit_reason == ProcessExitReason.COMPLETED:
        return _deserialize_function_result(raw_result)
    elif raw_result.exit_reason == ProcessExitReason.TIMEOUT:
        timeout_seconds = (job.timeout.total_seconds()
                           if job.timeout else "unknown")
        return FunctionResult(
            error_message=
            f"Timeout: Function execution exceeded {timeout_seconds} "
            f"seconds.\n{raw_result.error_message}")
    elif raw_result.exit_reason == ProcessExitReason.STARTUP_FAILURE:
        return FunctionResult(
            error_message=f"Startup failure: {raw_result.error_message}")
    else:
        raise ValueError(f"Unknown exit reason: {raw_result.exit_reason}")
