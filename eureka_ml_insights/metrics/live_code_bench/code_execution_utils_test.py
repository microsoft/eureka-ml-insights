"""Unit tests for code_execution_utils.py module.

To run:
    python -m unittest \
        eureka_ml_insights.metrics.live_code_bench.code_execution_utils_test
"""
import unittest
import subprocess
import datetime
import pickle

from eureka_ml_insights.metrics.live_code_bench import code_execution_utils


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
        self.job = code_execution_utils.FunctionJob(
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

        result = code_execution_utils.execute_function(
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

        result = code_execution_utils.execute_function(
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

        result = code_execution_utils.execute_function(
            job=self.job,
            runner=self.mock_runner,
        )

        self.assertFalse(result.success)
        self.assertRegex(result.error_message, "Startup failure:")


class ExecuteScriptTest(unittest.TestCase):

    def setUp(self):
        self.mock_runner = MockProcessRunner()
        self.job = code_execution_utils.ScriptJob(
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

        result = code_execution_utils.execute_script(job=self.job,
                                                     runner=self.mock_runner)

        self.assertEqual(result.stdout, expected_stdout)
        self.assertEqual(result.stderr, expected_stderr)
        self.assertEqual(result.error_message, "")

    def test_execute_script_timeout(self):
        """Test script execution that times out."""
        self.mock_runner.should_timeout = True

        result = code_execution_utils.execute_script(
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

        result = code_execution_utils.execute_script(
            job=self.job,
            runner=self.mock_runner,
        )

        self.assertRegex(result.error_message, "Startup failure:")


if __name__ == "__main__":
    unittest.main()
