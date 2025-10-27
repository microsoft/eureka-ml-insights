import datetime
import sys
import unittest

from eureka_ml_insights.core.job_runner.command_runners import base
from eureka_ml_insights.core.job_runner.command_runners import subprocess_runner


class TestSubprocessCommandRunner(unittest.TestCase):
    """Unit tests for SubprocessCommandRunner."""

    def test_run_success(self):
        runner = subprocess_runner.SubprocessCommandRunner()
        result = runner.run(["echo", "hello"])
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), b"hello")
        self.assertEqual(result.status, base.CommandStatus.COMPLETED)

    def test_run_timeout(self):
        runner = subprocess_runner.SubprocessCommandRunner()
        result = runner.run(
            ["sleep", "2"], timeout=datetime.timedelta(seconds=0.5))
        self.assertEqual(result.status, base.CommandStatus.TIMEOUT)
        self.assertEqual(result.returncode, -1)

    def test_run_invalid_command(self):
        runner = subprocess_runner.SubprocessCommandRunner()
        result = runner.run(["nonexistent_command"])
        self.assertEqual(result.status, base.CommandStatus.FAILED_TO_RUN)
        self.assertEqual(result.returncode, -1)
        self.assertTrue(result.stderr)

    def test_preexec_fn_called(self):
        def preexec_fn():
            sys.stdout.write("Preexec function called\n")
            sys.stdout.flush()

        runner = subprocess_runner.SubprocessCommandRunner(
            preexec_fn=preexec_fn)
        result = runner.run(["echo", "Echoing after preexec"])
        self.assertEqual(result.returncode, 0)
        self.assertEqual(
            result.stdout.strip(),
            b"Preexec function called\nEchoing after preexec")
        
    def test_stdin_passed(self):
        runner = subprocess_runner.SubprocessCommandRunner()
        result = runner.run(["cat"], stdin=b"input data")
        self.assertEqual(result.stdout, b"input data")
        self.assertEqual(result.status, base.CommandStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
