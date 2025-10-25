import unittest
import textwrap

from eureka_ml_insights.core.job_runner.command_runners import subprocess_runner
from eureka_ml_insights.metrics.live_code_bench import functional_test_case
from eureka_ml_insights.metrics.live_code_bench import functional_test_case_evaluator


class EvaluateFunctionalTestCase(unittest.TestCase):
    """Tests for the evaluate_functional_test_case function."""

    @classmethod
    def setUpClass(cls):
        """Sets up a command runner for the tests."""
        cls.command_runner = subprocess_runner.SubprocessCommandRunner()
    
    def test_evaluate_functional_case_passes(self):
        """Evaluates a functional test case that should pass."""
        src_code = textwrap.dedent("""
            def add(a, b):
                return a + b
        """)
        function_name = "add"
        test_case = functional_test_case.FunctionalTestCase(
            inputs=(3, 4),
            expected_output=7
        )

        result = functional_test_case_evaluator.evaluate_functional_test_case(
            src_code=src_code,
            runner=self.command_runner,
            function_name=function_name,
            test_case=test_case
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.error_message, "")

    def test_evaluate_functional_case_output_mismatch(self):
        """Evaluates a functional test case that should fail due to output mismatch."""
        src_code = textwrap.dedent("""
            def add(a, b):
                return a - b  # Incorrect implementation
        """)
        function_name = "add"
        test_case = functional_test_case.FunctionalTestCase(
            inputs=(3, 4),
            expected_output=7
        )

        result = functional_test_case_evaluator.evaluate_functional_test_case(
            src_code=src_code,
            runner=self.command_runner,
            function_name=function_name,
            test_case=test_case
        )

        self.assertFalse(result.passed)
        self.assertIn("Expected output: 7, but got: -1", result.error_message)
    
    def test_evaluate_functional_case_runtime_error(self):
        """Evaluates a functional test case that should fail due to runtime error."""
        src_code = textwrap.dedent("""
            def divide(a, b):
                return a / b
        """)
        function_name = "divide"
        test_case = functional_test_case.FunctionalTestCase(
            inputs=(3, 0),  # This will cause a division by zero error
            expected_output=None
        )

        result = functional_test_case_evaluator.evaluate_functional_test_case(
            src_code=src_code,
            runner=self.command_runner,
            function_name=function_name,
            test_case=test_case
        )

        self.assertFalse(result.passed)
        self.assertIn("division by zero", result.error_message)
    
    def test_evaluate_functional_case_missing_function_name_fails(self):
        """Evaluates a functional test case with missing function name."""
        src_code = textwrap.dedent("""
            def add(a, b):
                return a + b
        """)
        test_case = functional_test_case.FunctionalTestCase(
            inputs=(3, 4),
            expected_output=7
        )

        with self.assertRaisesRegex(
            ValueError,
            "function_name must be provided for FunctionalTestCase."
        ):
            functional_test_case_evaluator.evaluate_functional_test_case(
                src_code=src_code,
                runner=self.command_runner,
                function_name="",
                test_case=test_case
            )
