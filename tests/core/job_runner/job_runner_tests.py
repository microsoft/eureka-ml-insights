import dataclasses
import unittest

from eureka_ml_insights.core.job_runner.jobs import base as jobs_base
from eureka_ml_insights.core.job_runner.command_runners import base as command_runners_base
from eureka_ml_insights.core.job_runner import job_runner


class FakeCommandRunner(command_runners_base.CommandRunner):
    """A fake command runner for testing run_job."""

    def __init__(
        self,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        status: command_runners_base.CommandStatus = command_runners_base.
        CommandStatus.COMPLETED,
    ):
        self._stdout = stdout
        self._stderr = stderr
        self._returncode = returncode
        self._status = status
        self.last_command = None
        self.last_stdin = None

    def run(self,
            command: list[str],
            stdin: bytes | None = None,
            timeout=None):
        self.last_command = command
        self.last_stdin = stdin
        return command_runners_base.CommandResult(
            stdout=self._stdout,
            stderr=self._stderr,
            returncode=self._returncode,
            status=self._status,
        )


@dataclasses.dataclass
class DummyJob(jobs_base.Job):
    """A minimal job implementation for testing run_job."""

    command_called: bool = False

    def get_command(self) -> list[str]:
        self.command_called = True
        return ["dummy_command"]

    def serialize_input(self) -> bytes:
        return b"dummy input"

    def deserialize_result(self, stdout: bytes, stderr: bytes, retcode: int):
        return stdout.decode("utf-8")


class TestRunJob(unittest.TestCase):
    """Unit tests for the run_job function."""

    def test_run_job_success(self):
        fake_runner = FakeCommandRunner(stdout=b"success")
        job = DummyJob()
        result: job_runner.JobExecutionResult = job_runner.run_job(
            job, fake_runner)

        self.assertEqual(result.job_result, "success")
        self.assertEqual(result.runner_status,
                         command_runners_base.CommandStatus.COMPLETED)
        self.assertEqual(fake_runner.last_command, job.get_command())
        self.assertEqual(fake_runner.last_stdin, b"dummy input")

    def test_run_job_timeout(self):
        fake_runner = FakeCommandRunner(
            stdout=b"",
            stderr=b"timeout",
            returncode=-1,
            status=command_runners_base.CommandStatus.TIMEOUT,
        )
        job = DummyJob()
        result: job_runner.JobExecutionResult = job_runner.run_job(
            job, fake_runner)

        self.assertEqual(result.runner_status,
                         command_runners_base.CommandStatus.TIMEOUT)
        self.assertEqual(result.job_result, "")

    def test_run_job_failure(self):
        fake_runner = FakeCommandRunner(
            stdout=b"",
            stderr=b"error occurred",
            returncode=-1,
            status=command_runners_base.CommandStatus.FAILED_TO_RUN,
        )
        job = DummyJob()
        result: job_runner.JobExecutionResult = job_runner.run_job(
            job, fake_runner)

        self.assertEqual(result.runner_status,
                         command_runners_base.CommandStatus.FAILED_TO_RUN)
        self.assertEqual(result.job_result, "")


if __name__ == "__main__":
    unittest.main()
