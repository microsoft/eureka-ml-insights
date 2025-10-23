"""Configuration for sandbox resource limits and syscall filtering.

The SandboxConfig class is used to define resource limits and syscall filtering
for a sandboxed subprocess. It produces a pre-execution function that can be
run before executing the subprocess to enforce the constraints.

Typical usage:

    config = SandboxConfig(
        max_memory_bytes=100 * 1024**2,  # 100 MB
        max_fds=3,
        blocked_syscalls=frozenset({"connect", "socket"})
    )
"""

import dataclasses
import os
import resource
import sys

from collections.abc import Callable


@dataclasses.dataclass(frozen=True)
class SandboxConfig:
    """Configuration for restricting resource usage in a sandboxed environment.

    Attributes:
        max_cpu_seconds: Maximum CPU time in seconds.
        max_memory_bytes: Maximum memory usage in bytes.
        max_fds: Maximum number of file descriptors.
        max_procs: Maximum number of processes that can be created.
        blocked_syscalls: Set of system calls to block.
        allow_privileged: Whether to allow running as a privileged user.
    """
    max_memory_bytes: int | None = None
    max_fds: int | None = None
    blocked_syscalls: frozenset[str] = frozenset()
    allow_privileged: bool = False

    def to_preexec_fn(self) -> Callable[[], None]:
        """Generates a pre-execution function to apply the sandbox constraints.

        This function can be used with subprocess.Popen's preexec_fn parameter.

        Returns:
            A callable that applies the resource limits when called.
        """

        def preexec_fn() -> None:
            try:
                _apply_syscall_filter(self.blocked_syscalls)
            except Exception as e:
                _die(f"Sandbox failed to apply syscall filter: {str(e)}")

            try:
                _apply_resource_limits(
                    max_memory_bytes=self.max_memory_bytes,
                    max_fds=self.max_fds,
                )
            except Exception as e:
                _die(f"Sandbox failed to apply resource limits: {str(e)}")

        return preexec_fn


def _apply_resource_limits_unix(
    max_memory_bytes: int | None,
    max_fds: int | None,
) -> None:
    """Apply resource limits on Unix-like systems."""
    if max_memory_bytes is not None:
        resource.setrlimit(resource.RLIMIT_AS,
                           (max_memory_bytes, max_memory_bytes))
    if max_fds is not None:
        resource.setrlimit(resource.RLIMIT_NOFILE, (max_fds, max_fds))


def _apply_resource_limits(
    max_memory_bytes: int | None,
    max_fds: int | None,
) -> None:
    """Apply resource limits in a platform-aware way."""
    platform: str = sys.platform

    if platform.startswith("linux") or platform.startswith("darwin"):
        _apply_resource_limits_unix(
            max_memory_bytes=max_memory_bytes,
            max_fds=max_fds,
        )
    else:
        raise NotImplementedError(
            f"Resource limiting not implemented for platform: {platform}")


def _apply_syscall_filter_linux(blocked_syscalls: frozenset[str]) -> None:
    """Apply syscall filtering on Linux using seccomp."""
    try:
        import seccomp  # type: ignore
    except ImportError:
        try:
            import pyseccomp as seccomp  # type: ignore
        except ImportError:
            raise RuntimeError(
                "seccomp is required for syscall filtering on Linux")

    # Default allow all syscalls
    filt = seccomp.SyscallFilter(defaction=seccomp.ALLOW)

    for syscall_name in blocked_syscalls:
        syscall_no: int = seccomp.resolve_syscall(
            seccomp.system_arch(), syscall_name)
        if syscall_no < 0:
            raise ValueError(f"Unknown syscall name: {syscall_name}")
        filt.add_rule(seccomp.KILL_PROCESS, syscall_name)

    filt.load()


def _apply_syscall_filter(blocked_syscalls: frozenset[str]) -> None:
    """Apply syscall filtering in a platform-aware way."""
    if not blocked_syscalls:
        return

    platform: str = sys.platform

    if platform.startswith("linux"):
        _apply_syscall_filter_linux(blocked_syscalls)
    else:
        raise NotImplementedError(
            f"Syscall filtering not implemented for platform: {platform}")


def _die(msg: str) -> None:
    """Write message to stderr and exit immediately.

    WARNING: Only safe to use in subprocess preexec_fn.
    """
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()
    os._exit(1)
