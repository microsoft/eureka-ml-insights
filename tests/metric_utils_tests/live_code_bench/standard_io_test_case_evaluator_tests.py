import unittest
import textwrap

from parameterized import parameterized

from eureka_ml_insights.core.job_runner.command_runners import subprocess_runner
from eureka_ml_insights.metrics.live_code_bench import standard_io_test_case
from eureka_ml_insights.metrics.live_code_bench import standard_io_test_case_evaluator


class EvaluateStandardIOTestCase(unittest.TestCase):
    """Tests for the evaluate_standard_io_test_case function."""

    @classmethod
    def setUpClass(cls):
        """Sets up a command runner for the tests."""
        cls.command_runner = subprocess_runner.SubprocessCommandRunner()
    
    @parameterized.expand([
        # (src_code, stdin, expected_stdout)

        # Simple addition
        (
            textwrap.dedent("""
                x = int(input())
                y = int(input())
                print(x + y)
            """),
            "2\n3\n",
            "5\n"
        ),

        # No newline in output
        (
            textwrap.dedent("""
                import sys

                x = int(input())
                y = int(input())

                sys.stdout.write(str(x + y))
            """),
            "2\n3\n",
            "5\n"
        ),
    ])
    def test_evaluate_stdin_case_passes(
        self, src_code: str, stdin: str, expected_stdout: str):
        """Evaluates a standard I/O test case that should pass."""
        test_case = standard_io_test_case.StandardIOTestCase(
            stdin=stdin,
            expected_stdout=expected_stdout
        )

        result = standard_io_test_case_evaluator.evaluate_standard_io_test_case(
            src_code=src_code,
            test_case=test_case,
            runner=self.command_runner,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.error_message, "")
    
    def test_evaluate_stdin_case_output_mismatch(self):
        """Evaluates a standard I/O test case that should fail due to output mismatch."""
        src_code = textwrap.dedent("""
            x = int(input())
            y = int(input())
            print(x * y)
        """)
        test_case = standard_io_test_case.StandardIOTestCase(
            stdin="2\n3\n",
            expected_stdout="a\n"
        )

        result = standard_io_test_case_evaluator.evaluate_standard_io_test_case(
            src_code=src_code,
            test_case=test_case,
            runner=self.command_runner,
        )

        self.assertFalse(result.passed)
        self.assertIn("Expected stdout: a, but got: 6", result.error_message)
    
    def test_evaluate_stdin_case_runtime_error(self):
        """Evaluates a standard I/O test case that should fail due to runtime error."""
        src_code = textwrap.dedent("""
            x = int(input())
            y = int(input())
            print(x / y)
        """)
        test_case = standard_io_test_case.StandardIOTestCase(
            stdin="5\n0\n",  # This will cause a division by zero error
            expected_stdout=""
        )

        result = standard_io_test_case_evaluator.evaluate_standard_io_test_case(
            src_code=src_code,
            test_case=test_case,
            runner=self.command_runner,
        )

        self.assertFalse(result.passed)
        self.assertIn("division by zero", result.error_message)
