"""Test for sandbox_config.py.

To run:
    python -m unittest \
        eureka_ml_insights.metrics.live_code_bench.sandbox_config_test
"""

import subprocess
import sys
import textwrap
import unittest

from parameterized import parameterized

from eureka_ml_insights.metrics.live_code_bench import sandbox_config


@unittest.skipIf(
    not sys.platform.startswith("linux"),
    "Resource limiting tests are only supported on Linux systems.")
class ResourceLimitIntegrationTest(unittest.TestCase):
    """Integration tests using actual subprocesses."""

    def test_memory_limit_prevents_large_allocation(self):
        """Test that memory limits prevent allocating too much memory."""
        config = sandbox_config.SandboxConfig(
            max_memory_bytes=100 * 1024 * 1024,  # Limit to 100MB
        )

        # Try to allocate 101MB
        code = "data = bytearray(100 * 1024 * 1024 + 1)"

        result = subprocess.run([sys.executable, '-c', code],
                                preexec_fn=config.to_preexec_fn(),
                                capture_output=True)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MemoryError", result.stderr.decode())

    def test_memory_limit_allows_small_allocation(self):
        """Test that memory limits allow allocating within the limit."""
        config = sandbox_config.SandboxConfig(
            max_memory_bytes=100 * 1024 * 1024,  # Limit to 100MB
        )

        # Try to allocate 50MB
        code = "data = bytearray(50 * 1024 * 1024)"

        result = subprocess.run([sys.executable, '-c', code],
                                preexec_fn=config.to_preexec_fn(),
                                capture_output=True)

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr.decode(), "")

    def test_fds_limit_prevents_opening_too_many_files(self):
        """Test that file descriptor limits prevent opening too many files."""
        config = sandbox_config.SandboxConfig(
            max_fds=10  # Limit to 10 file descriptors
        )

        # Try to open 11 files
        code = textwrap.dedent("""
            files = []
            for i in range(11):
                files.append(open('/dev/null', 'r'))
        """)

        result = subprocess.run([sys.executable, '-c', code],
                                preexec_fn=config.to_preexec_fn(),
                                capture_output=True)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Too many open files", result.stderr.decode())

    def test_fds_limit_allows_opening_within_limit(self):
        """Test that file descriptor limits allow opening files within the limit."""
        config = sandbox_config.SandboxConfig(
            max_fds=10  # Limit to 10 file descriptors
        )

        # Try to open 5 files
        code = textwrap.dedent("""
            files = []
            for i in range(5):
                files.append(open('/dev/null', 'r'))
        """)

        result = subprocess.run([sys.executable, '-c', code],
                                preexec_fn=config.to_preexec_fn(),
                                capture_output=True)

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr.decode(), "")


@unittest.skipIf(
    not sys.platform.startswith("linux"),
    "Syscall filtering tests are only supported on Linux systems.")
class SyscallFilterIntegrationTest(unittest.TestCase):
    """Integration tests to verify syscall filtering works."""

    @parameterized.expand([
        # (blocked_syscalls, code, should_succeed, stderr_contains)

        # Test blocking network connect syscall
        (
            frozenset({"socket"}),
            textwrap.dedent("""
                    import socket
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect(('localhost', 0))
                """),
            False,
        ),

        # Test allowing network connect syscall
        (
            frozenset({"fork"}),
            textwrap.dedent("""
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(('localhost', 0))
            """),
            True,
        ),

        # Test blocking file open syscall
        (
            frozenset({"open", "openat"}),
            textwrap.dedent("""
                f = open('/dev/null', 'r')
                data = f.read()
            """),
            False,
        ),

        # Test allowing file open syscall
        (
            frozenset({"socket"}),
            textwrap.dedent("""
                f = open('/dev/null', 'r')
                data = f.read()
            """),
            True,
        ),

        # Test blocking process creation syscall
        (
            frozenset({"fork", "vfork", "clone"}),
            textwrap.dedent("""
                import os
                pid = os.fork()
                if pid == 0:
                    os._exit(0)
                else:
                    os.waitpid(pid, 0)
            """),
            False,
        ),

        # Test allowing process creation syscall
        (
            frozenset({"socket"}),
            textwrap.dedent("""
                import os
                pid = os.fork()
                if pid == 0:
                    os._exit(0)
                else:
                    os.waitpid(pid, 0)
            """),
            True,
        ),
    ])
    def test_syscall_behavior(self, blocked_syscalls: frozenset[str],
                              code: str, should_succeed: bool) -> None:
        """Test syscall filtering behavior parameterized by syscall and expected outcome."""
        config = sandbox_config.SandboxConfig(
            blocked_syscalls=blocked_syscalls)
        code = textwrap.dedent(code)

        result = subprocess.run([sys.executable, "-c", code],
                                preexec_fn=config.to_preexec_fn(),
                                capture_output=True)

        if should_succeed:
            self.assertEqual(result.returncode, 0)
        else:
            self.assertNotEqual(result.returncode, 0)

    def test_unknown_syscall_name_raises_error(self):
        """Test that an unknown syscall name raises an error in the subprocess."""
        config = sandbox_config.SandboxConfig(
            blocked_syscalls=frozenset({"nonexistent_syscall"}))

        result = subprocess.run([sys.executable, "-c", "pass"],
                                preexec_fn=config.to_preexec_fn(),
                                capture_output=True)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "Unknown syscall name", result.stderr.decode())


if __name__ == '__main__':
    unittest.main()
