"""Defines how to execute a Python function from a source script.

The function should be defined in the provided source script, which is expected
to be self-contained, including all necessary imports and definitions.
The function must return a picklable object.
"""

import dataclasses
import pickle
import sys
import textwrap

from typing import Any

from eureka_ml_insights.core.job_runner.jobs import base


# Minimal wrapper script to run functions.
# This script reads a pickled FunctionJob from stdin, executes the function,
# and writes the pickled result to stdout.
_FUNCTION_RUNNER_SCRIPT = textwrap.dedent("""\
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

src_script = input_data["src_script"]
function_name = input_data["function_name"]
args = input_data["args"]
kwargs = input_data["kwargs"]

code_object = compile(src_script, '<my_source>', 'exec')

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
        "result": result,
        "exception_class_name": None,
        "exception_msg": None,
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr_buffer.getvalue(),
    }
except Exception as e:
    output = {
        "result": None,
        "exception_class_name": e.__class__.__name__,
        "exception_msg": traceback.format_exc(),
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr_buffer.getvalue(),
    }

# Write *only the pickle* to stdout.buffer
try:
    pickle.dump(output, sys.stdout.buffer)
except Exception as e:
    fallback_output = {
        "result": None,
        "exception_class_name": e.__class__.__name__,
        "exception_msg": f"Pickle error: {traceback.format_exc()}",
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr_buffer.getvalue(),
    }
    pickle.dump(fallback_output, sys.stdout.buffer)
""")

@dataclasses.dataclass(frozen=True)
class PythonFunctionFromScriptJobResult:
    """Result of executing a PythonFunctionFromScriptJob.

    Attributes:
        return_value: The return value of the executed function.
        stdout: Captured standard output during execution.
        stderr: Captured standard error during execution.
        exception_class_name: Name of the exception class if an exception was
            raised, else None.
        exception_msg: The exception message and traceback if an exception was
            raised, else None.
    """
    return_value: Any = None
    stdout: str = ""
    stderr: str = ""
    exception_class_name: str | None = None
    exception_msg: str | None = None

    @property
    def success(self) -> bool:
        """Returns True if execution completed without errors."""
        return self.exception_class_name is None


@dataclasses.dataclass(frozen=True)
class PythonFunctionFromScriptJob(base.Job):
    """Job to execute a Python function defined in a source script.

    The function must return a picklable object. The source script should be
    self-contained, including all necessary imports and definitions.

    Attributes:
        src_script: The source code of the Python script as a string.
        function_name: The name of the function to execute. If the function is
            a method of a class, use the dotted path (e.g.,
            'MyClass.my_method'). The class will be instantiated with no
            arguments and the method called on the instance.
        args: Positional arguments to pass to the function.
        kwargs: Keyword arguments to pass to the function.
    """
    src_script: str
    function_name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def get_command(self) -> list[str]:
        """Returns the command to execute the job in the command line."""
        return [sys.executable, "-c", _FUNCTION_RUNNER_SCRIPT]

    def serialize_input(self) -> bytes:
        """Serializes the input needed to execute the job."""
        input_data = {
            "src_script": self.src_script,
            "function_name": self.function_name,
            "args": self.args,
            "kwargs": self.kwargs,
        }

        try:
            return pickle.dumps(input_data)
        except Exception as e:
            raise ValueError(f"Failed to serialize job input") from e

    def deserialize_result(
        self,
        stdout: bytes,
        stderr: bytes,
        retcode: int
    ) -> PythonFunctionFromScriptJobResult:
        """Deserializes the job result from the runner output.

        Args:
            stdout: The standard output from the job runner.
            stderr: The standard error from the job runner.
            retcode: The return code from the job runner.

        Returns:
            The deserialized PythonFunctionFromScriptJobResult.

        Raises:
            ValueError: If the result cannot be deserialized or does not have
                the expected format.
        """
        try:
            result = pickle.loads(stdout)
        except Exception as e:
            raise ValueError(f"Failed to deserialize job result: {e}")

        if not isinstance(result, dict):
            raise ValueError(
                "Invalid result format from job runner. Expected dict "
                f"but got {type(result).__name__}")

        expected_keys: frozenset[str] = frozenset({
            "result", "exception_class_name",
            "exception_msg", "stdout", "stderr"
        })

        missing_keys: frozenset[str] = expected_keys - set(result.keys())

        if missing_keys:
            raise ValueError(
                "Missing keys in result from job runner: "
                f"{', '.join(missing_keys)}")

        return PythonFunctionFromScriptJobResult(
            return_value=result["result"],
            stdout=result["stdout"],
            stderr=result["stderr"],
            exception_class_name=result["exception_class_name"],
            exception_msg=result["exception_msg"]
        )
