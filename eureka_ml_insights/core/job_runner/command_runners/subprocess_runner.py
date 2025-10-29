"""Defines a command runner that uses subprocess to execute commands."""

import subprocess
import datetime

from typing import Callable, TypeAlias

from eureka_ml_insights.core.job_runner.command_runners import base

_PreexecFn: TypeAlias = Callable[[], None]


class SubprocessCommandRunner(base.CommandRunner):
    """Command runner that uses subprocess to execute commands."""

    def __init__(self, preexec_fn: _PreexecFn | None = None) -> None:
        self._preexec_fn = preexec_fn

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
                preexec_fn=self._preexec_fn,
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
