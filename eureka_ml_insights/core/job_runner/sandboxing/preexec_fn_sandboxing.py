"""Configuration for sandbox resource limits and syscall filtering.

Produces a preexec_fn suitable for subprocess.Popen to apply the sandboxing.
"""

import dataclasses
import os
import resource
import sys
from typing import Callable, TypeAlias

_PreexecFn: TypeAlias = Callable[[], None]


@dataclasses.dataclass
class SandboxConfig:
    """Configuration for restricting resource usage in a sandboxed environment.

    Attributes:
        max_memory_bytes: Maximum memory usage in bytes.
        max_fds: Maximum number of file descriptors.
        blocked_syscalls: Set of system calls to block (Linux only).
        allow_privileged: Whether to allow running as a privileged user.
    """
    max_memory_bytes: int | None = None
    blocked_syscalls: set[str] = dataclasses.field(default_factory=set)

    def to_preexec_fn(self) -> _PreexecFn:
        """Generates a pre-execution function to apply the sandbox constraints.

        Returns:
            A callable suitable for `subprocess.Popen(preexec_fn=...)`.
        """

        def preexec_fn() -> None:
            try:
                _apply_syscall_filter_linux_only(self.blocked_syscalls)
            except Exception as e:
                _die(f"Sandbox failed to apply syscall filter: {str(e)}")

            try:
                _apply_resource_limits_posix(
                    max_memory_bytes=self.max_memory_bytes, )
            except Exception as e:
                _die(f"Sandbox failed to apply resource limits: {str(e)}")

        return preexec_fn


def _apply_resource_limits_posix(max_memory_bytes: int | None) -> None:
    """Apply resource limits on POSIX systems (Linux/macOS)."""
    platform = sys.platform
    if not (platform.startswith("linux") or platform.startswith("darwin")):
        raise NotImplementedError(
            f"Resource limiting not supported on platform: {platform}")

    if max_memory_bytes is not None:
        resource.setrlimit(
            resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))


def _apply_syscall_filter_linux_only(blocked_syscalls: set[str]) -> None:
    """Apply syscall filtering on Linux using seccomp. Default allow all."""
    if not blocked_syscalls:
        return

    platform = sys.platform
    if not platform.startswith("linux"):
        raise NotImplementedError(
            f"Syscall filtering only supported on Linux. Platform: {platform}")

    try:
        import seccomp  # type: ignore
    except ImportError:
        try:
            import pyseccomp as seccomp  # type: ignore
        except ImportError:
            raise RuntimeError(
                "seccomp is required for syscall filtering on Linux")

    filt = seccomp.SyscallFilter(defaction=seccomp.ALLOW)

    for syscall_name in blocked_syscalls:
        syscall_no = seccomp.resolve_syscall(
            seccomp.system_arch(), syscall_name)
        if syscall_no < 0:
            raise ValueError(f"Unknown syscall name: {syscall_name}")
        filt.add_rule(seccomp.KILL_PROCESS, syscall_name)

    filt.load()


def _die(msg: str) -> None:
    """Write message to stderr and exit immediately."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()
    os._exit(1)
