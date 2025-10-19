"""Unit tests for evaluate_codegen.py.

To run:
    python -m unittest \
        eureka_ml_insights.metrics.live_code_bench.evaluate_codegen_test
"""

import unittest
import textwrap

from parameterized import parameterized
from typing import Any

from eureka_ml_insights.metrics.live_code_bench import evaluate_codegen


class ParseTestCaseTest(unittest.TestCase):
    """Tests for the parse_test_case function."""

    @parameterized.expand([
        ("5", "10", (5,), 10),
        ("(1, 2)", "3", (1, 2), 3),
        ("('a', 'b', 'c')", "'abc'", ("a", "b", "c"), "abc"),
        ("[]", "[]", ([],), []),
        ("(1, 'a', 3.5)", "(2, 'b', 4.5)", (1, 'a', 3.5), (2, 'b', 4.5)),
        (
            "[[1, 2], [3, 4]]",
            "[[5, 6], [7, 8]]",
            ([[1, 2], [3, 4]],),
            [[5, 6], [7, 8]]
        ),
    ])
    def test_parse_functional_case(
        self, inputs: str, output: str, expected_inputs: tuple[Any, ...],
        expected_output: Any):
        """Parses a functional test case dictionary correctly."""
        data = {
            "testtype": "functional",
            "inputs": inputs,
            "output": output,
        }

        result = evaluate_codegen.parse_test_case(data)

        self.assertIsInstance(result, evaluate_codegen.FunctionalTestCase)
        self.assertEqual(result.inputs, expected_inputs)  # type: ignore
        self.assertEqual(result.expected_output, expected_output)  # type: ignore
    
    def test_parse_stdin_case(self):
        """Parses a standard I/O test case dictionary correctly."""
        data = {
            "testtype": "stdin",
            "input": "5\n6",
            "output": "6\n",
        }

        result = evaluate_codegen.parse_test_case(data)

        self.assertIsInstance(result, evaluate_codegen.StandardIOTestCase)
        self.assertEqual(result.stdin, "5\n6")  # type: ignore
        self.assertEqual(result.expected_stdout, "6\n")  # type: ignore

    def test_invalid_testtype_raises(self):
        """Raises ValueError for an unknown test type."""
        data = {
            "testtype": "unknown",
            "input": "a",
            "output": "b",
        }

        with self.assertRaisesRegex(ValueError, "Unknown test type"):
            evaluate_codegen.parse_test_case(data)


class EvaluateTestCaseTest(unittest.TestCase):
    """Tests for the evaluate_codegen function."""

    def test_evaluate_functional_case_passes(self):
        """Evaluates a functional test case that should pass."""
        src_code = textwrap.dedent("""
            def add(a, b):
                return a + b
        """)
        function_name = "add"
        test_case = evaluate_codegen.FunctionalTestCase(
            inputs=(3, 4),
            expected_output=7
        )

        result = evaluate_codegen.evaluate_test_case(
            src_code=src_code,
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
        test_case = evaluate_codegen.FunctionalTestCase(
            inputs=(3, 4),
            expected_output=7
        )

        result = evaluate_codegen.evaluate_test_case(
            src_code=src_code,
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
        test_case = evaluate_codegen.FunctionalTestCase(
            inputs=(3, 0),  # This will cause a division by zero error
            expected_output=None
        )

        result = evaluate_codegen.evaluate_test_case(
            src_code=src_code,
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
        test_case = evaluate_codegen.FunctionalTestCase(
            inputs=(3, 4),
            expected_output=7
        )

        with self.assertRaisesRegex(
            ValueError,
            "function_name must be provided for FunctionalTestCase."
        ):
            evaluate_codegen.evaluate_test_case(
                src_code=src_code,
                function_name="",
                test_case=test_case
            )
    
    def test_evaluate_stdin_case_passes(self):
        """Evaluates a standard I/O test case that should pass."""
        src_code = textwrap.dedent("""
            x = int(input())
            y = int(input())
            print(x + y)
        """)
        test_case = evaluate_codegen.StandardIOTestCase(
            stdin="1\n2\n",
            expected_stdout="3\n"
        )

        result = evaluate_codegen.evaluate_test_case(
            src_code=src_code,
            test_case=test_case
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
        test_case = evaluate_codegen.StandardIOTestCase(
            stdin="2\n3\n",
            expected_stdout="a\n"
        )

        result = evaluate_codegen.evaluate_test_case(
            src_code=src_code,
            test_case=test_case
        )

        self.assertFalse(result.passed)
        self.assertIn("Expected stdout: a\n, but got: 6\n", result.error_message)
    
    def test_evaluate_stdin_case_runtime_error(self):
        """Evaluates a standard I/O test case that should fail due to runtime error."""
        src_code = textwrap.dedent("""
            x = int(input())
            y = int(input())
            print(x / y)
        """)
        test_case = evaluate_codegen.StandardIOTestCase(
            stdin="5\n0\n",  # This will cause a division by zero error
            expected_stdout=""
        )

        result = evaluate_codegen.evaluate_test_case(
            src_code=src_code,
            test_case=test_case
        )

        self.assertFalse(result.passed)
        self.assertIn("division by zero", result.error_message)
    

if __name__ == "__main__":
    unittest.main()
