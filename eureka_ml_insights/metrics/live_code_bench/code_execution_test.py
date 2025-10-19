"""Unit tests for code_execution_utils.py module.

To run:
    python -m unittest \
        eureka_ml_insights.metrics.live_code_bench.code_execution_test
"""

import datetime
import pickle
import subprocess
import textwrap
import unittest

from parameterized import parameterized

from eureka_ml_insights.metrics.live_code_bench import code_execution


class MockProcessRunner:
    """Mock ProcessRunner for testing.

    Attributes:
        stdout: Bytes to return as stdout.
        stderr: Bytes to return as stderr.
        returncode: The return code of the process.
        should_timeout: If True, simulates a timeout when run() is called.
        should_fail_startup: If True, simulates a startup failure when run() is
            called
    """

    def __init__(self):
        self.stdout = b""
        self.stderr = b""
        self.returncode = 0
        self.should_timeout = False
        self.should_fail_startup = False

    def run(self, args: list[str], input: bytes | None, capture_output: bool,
            timeout: datetime.timedelta | None) -> subprocess.CompletedProcess:
        if self.should_timeout and timeout is not None:
            raise subprocess.TimeoutExpired(args, timeout.total_seconds())
        elif self.should_timeout:
            raise ValueError(
                "Timeout must be specified if should_timeout is True")

        if self.should_fail_startup:
            raise RuntimeError("Mock startup failure")

        return subprocess.CompletedProcess(args=args,
                                           returncode=self.returncode,
                                           stdout=self.stdout,
                                           stderr=self.stderr)


class ExecuteFunctionTest(unittest.TestCase):

    def setUp(self):
        self.mock_runner = MockProcessRunner()
        self.job = code_execution.FunctionJob(
            src_code="def add(a, b): return a + b",
            function_name="add",
            args=(2, 3),
            timeout=datetime.timedelta(seconds=5))

    def test_execute_function_success(self):
        """Test successful function execution."""
        expected_result = 5
        expected_stdout = "Something printed"
        expected_stderr = "Some logging info"

        self.mock_runner.stdout = pickle.dumps({
            "success": True,
            "result": expected_result,
            "error": "",
            "stdout": expected_stdout,
            "stderr": expected_stderr,
        })

        result = code_execution.execute_function(
            job=self.job,
            runner=self.mock_runner,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.return_value, expected_result)
        self.assertEqual(result.error_message, "")
        self.assertEqual(result.stdout, expected_stdout)
        self.assertEqual(result.stderr, expected_stderr)

    def test_execute_function_timeout(self):
        """Test function execution that times out."""
        self.mock_runner.should_timeout = True

        result = code_execution.execute_function(
            job=self.job,
            runner=self.mock_runner,
        )

        self.assertFalse(result.success)

        assert self.job.timeout is not None
        self.assertRegex(
            result.error_message, "Timeout: Function execution exceeded "
            f"{self.job.timeout.total_seconds()} seconds.")

    def test_execute_function_startup_failure(self):
        """Test function execution that fails to start."""

        self.mock_runner.should_fail_startup = True

        result = code_execution.execute_function(
            job=self.job,
            runner=self.mock_runner,
        )

        self.assertFalse(result.success)
        self.assertRegex(result.error_message, "Startup failure:")


class ExecuteScriptTest(unittest.TestCase):

    def setUp(self):
        self.mock_runner = MockProcessRunner()
        self.job = code_execution.ScriptJob(
            script="import sys; print(int(sys.stdin.read()) + 1)",
            stdin_input="5\n",
            timeout=datetime.timedelta(seconds=5))

    def test_execute_script_success(self):
        """Test successful script execution."""
        expected_stdout = "6\n"
        expected_stderr = "Script logging"

        self.mock_runner.stdout = expected_stdout.encode("utf-8")
        self.mock_runner.stderr = expected_stderr.encode("utf-8")
        self.mock_runner.returncode = 0

        result = code_execution.execute_script(job=self.job,
                                                     runner=self.mock_runner)

        self.assertEqual(result.stdout, expected_stdout)
        self.assertEqual(result.stderr, expected_stderr)
        self.assertEqual(result.error_message, "")

    def test_execute_script_timeout(self):
        """Test script execution that times out."""
        self.mock_runner.should_timeout = True

        result = code_execution.execute_script(
            job=self.job,
            runner=self.mock_runner,
        )

        assert self.job.timeout is not None
        self.assertRegex(
            result.error_message, "Timeout: Script execution exceeded "
            f"{self.job.timeout.total_seconds()} seconds.")

    def test_execute_script_startup_failure(self):
        """Test script execution that fails to start."""
        self.mock_runner.should_fail_startup = True

        result = code_execution.execute_script(
            job=self.job,
            runner=self.mock_runner,
        )

        self.assertRegex(result.error_message, "Startup failure:")


class ExecuteFunctionIntegrationTest(unittest.TestCase):
    """Integration tests for execute_function that actually runs subprocesses."""

    @parameterized.expand([
        (
            code_execution.FunctionJob(
                src_code=textwrap.dedent("""\
                        import sys
                        import math

                        def multiply_and_sqrt(a, b):
                            print("Running multiply_and_sqrt")
                            print("Logging info", file=sys.stderr)
                            return math.sqrt(a * b)
                    """),
                function_name="multiply_and_sqrt",
                args=(16, 9),
            ),
            code_execution.FunctionResult(
                return_value=12.0,
                stdout="Running multiply_and_sqrt\n",
                stderr="Logging info\n",
            ),
        ),
        (
            code_execution.FunctionJob(
                src_code=textwrap.dedent("""\
                    _CONSTANT = 10

                    def foo() -> int:
                        print("In foo")
                        return 42 + _CONSTANT

                    def bar(a: str) -> tuple[int, str]:
                        print("In bar")
                        return foo(), a
                """),
                function_name="bar",
                kwargs={"a": "test"},
            ),
            code_execution.FunctionResult(
                return_value=(52, "test"),
                error_message="",
                stdout="In bar\nIn foo\n",
            ),
        ),
        (
            code_execution.FunctionJob(
                src_code=textwrap.dedent("""\
                    class Solution:
                        def add(self, x: int, y: int) -> int:
                            print("Adding numbers")
                            return x + y
                """),
                function_name="Solution.add",
                args=(5, 7),
            ),
            code_execution.FunctionResult(
                return_value=12,
                stdout="Adding numbers\n",
            ),
        ),
    ])
    def test_success(self, job: code_execution.FunctionJob,
                     expected_result: code_execution.FunctionResult):
        """Test happy path function execution."""
        result = code_execution.execute_function(job)

        self.assertEqual(result.success, True)
        self.assertEqual(result.return_value, expected_result.return_value)
        self.assertEqual(result.error_message, expected_result.error_message)
        self.assertEqual(result.stdout, expected_result.stdout)
        self.assertEqual(result.stderr, expected_result.stderr)

    def test_timeout(self):
        """Test function execution that times out."""
        job = code_execution.FunctionJob(
            src_code=textwrap.dedent("""\
                import time

                def long_running_function():
                    time.sleep(10)
                    return "Done"
            """),
            function_name="long_running_function",
            timeout=datetime.timedelta(seconds=1),
        )

        result = code_execution.execute_function(job)

        self.assertFalse(result.success)
        self.assertIsNone(result.return_value)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")

        assert job.timeout is not None
        self.assertRegex(
            result.error_message, "Timeout: Function execution exceeded "
            f"{job.timeout.total_seconds()} seconds.")

    def test_error_in_function(self):
        """Test function execution that raises an error."""
        job = code_execution.FunctionJob(
            src_code=textwrap.dedent("""\
                def faulty_function():
                    print("This will fail")
                    raise ValueError("Intentional error")
            """),
            function_name="faulty_function",
        )

        result = code_execution.execute_function(job)

        self.assertFalse(result.success)
        self.assertIsNone(result.return_value)
        self.assertEqual(result.stdout, "This will fail\n")
        self.assertEqual(result.stderr, "")
        self.assertRegex(result.error_message, "ValueError: Intentional error")
    
    def test_syntax_error_in_code(self):
        """Test function execution with syntax error in source code."""
        job = code_execution.FunctionJob(
            src_code=textwrap.dedent("""\
                def broken_function()
                    return "Missing colon"
            """),
            function_name="broken_function",
        )

        result = code_execution.execute_function(job)

        self.assertFalse(result.success)
        self.assertIsNone(result.return_value)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        self.assertRegex(result.error_message, "SyntaxError")
    
    def test_function_not_found(self):
        """Test function execution when the specified function is not found."""
        job = code_execution.FunctionJob(
            src_code=textwrap.dedent("""\
                def existing_function():
                    return "I exist"
            """),
            function_name="non_existent_function",
        )

        result = code_execution.execute_function(job)

        self.assertFalse(result.success)
        self.assertIsNone(result.return_value)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        self.assertRegex(
            result.error_message,
            "'non_existent_function' not found")


class ExecuteScriptIntegrationTest(unittest.TestCase):
    """Integration tests for execute_script that actually runs subprocesses."""

    @parameterized.expand([
        (
            code_execution.ScriptJob(
                script=textwrap.dedent("""\
                    import sys
                    data = sys.stdin.read()
                    print(f"Received: {data.strip()}")
                    print("Some debug info", file=sys.stderr)
                """),
                stdin_input="Hello, World!\n",
            ),
            code_execution.ScriptResult(
                stdout="Received: Hello, World!\n",
                stderr="Some debug info\n",
            ),
        ),
        (
            code_execution.ScriptJob(
                script=textwrap.dedent("""\
                    import math

                    def crazy_add(a, b):
                        return int(math.pow(a + b, 2))

                    def main():
                        x = int(input("Enter a number: "))
                        y = int(input("Enter another number: "))
                        print(crazy_add(x, y))

                    if __name__ == "__main__":
                        main()
                """),
                stdin_input="3\n4\n",
            ),
            code_execution.ScriptResult(
                stdout="Enter a number: Enter another number: 49\n",
            ),
        ),
    ])
    def test_success(self, job: code_execution.ScriptJob,
                     expected_result: code_execution.ScriptResult):
        """Test happy path script execution."""
        result = code_execution.execute_script(job)

        self.assertEqual(result.stdout, expected_result.stdout)
        self.assertEqual(result.stderr, expected_result.stderr)
        self.assertEqual(result.error_message, "")
    
    def test_timeout(self):
        """Test script execution that times out."""
        job = code_execution.ScriptJob(
            script=textwrap.dedent("""\
                import time
                time.sleep(10)
                print("Done")
            """),
            timeout=datetime.timedelta(seconds=1),
        )

        result = code_execution.execute_script(job)

        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")

        assert job.timeout is not None
        self.assertRegex(
            result.error_message, "Timeout: Script execution exceeded "
            f"{job.timeout.total_seconds()} seconds.")
    
    def test_error_in_script(self):
        """Test script execution that raises an error."""
        job = code_execution.ScriptJob(
            script=textwrap.dedent("""\
                print("About to fail")
                raise RuntimeError("Intentional script error")
            """),
        )

        result = code_execution.execute_script(job)

        self.assertEqual(result.stdout, "About to fail\n")
        self.assertRegex(
            result.stderr, "RuntimeError: Intentional script error")
        self.assertRegex(
            result.error_message, "RuntimeError: Intentional script error")
    
    def test_syntax_error_in_script(self):
        """Test script execution with syntax error in script."""
        job = code_execution.ScriptJob(
            script=textwrap.dedent("""\
                def broken_function()
                    print("I am missing a colon")
            """),
        )

        result = code_execution.execute_script(job)

        self.assertEqual(result.stdout, "")
        self.assertRegex(result.stderr, "SyntaxError")
        self.assertRegex(result.error_message, "SyntaxError")

if __name__ == "__main__":
    unittest.main()
