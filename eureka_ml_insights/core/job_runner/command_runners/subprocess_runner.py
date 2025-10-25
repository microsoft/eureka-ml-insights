"""Defines a command runner that uses subprocess to execute commands."""

import subprocess
import datetime

from eureka_ml_insights.core.job_runner.command_runners import base


class SubprocessCommandRunner(base.CommandRunner):
    """Command runner that uses subprocess to execute commands."""

    def run(
        self,
        command: list[str],
        stdin: bytes | None = None,
        timeout: datetime.timedelta | None = None,
    ) -> base.CommandResult:
        try:
            completed = subprocess.run(
                command,
                input=stdin,
                capture_output=True,
                timeout=timeout.total_seconds() if timeout else None,
                check=False,
            )
            return base.CommandResult(
                stdout=completed.stdout,
                stderr=completed.stderr,
                returncode=completed.returncode,
                status=base.CommandStatus.COMPLETED,
            )
        except subprocess.TimeoutExpired as e:
            return base.CommandResult(
                stdout=e.stdout or b"",
                stderr=e.stderr or str(e).encode("utf-8"),
                returncode=-1,
                status=base.CommandStatus.TIMEOUT,
            )
        except Exception as e:
            return base.CommandResult(
                stdout=b"",
                stderr=str(e).encode("utf-8"),
                returncode=-1,
                status=base.CommandStatus.FAILED_TO_RUN,
            )
