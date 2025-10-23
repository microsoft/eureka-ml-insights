import unittest
import textwrap

from parameterized import parameterized
from typing import Any

from eureka_ml_insights.metrics.live_code_bench import evaluate_codegen


class ParseTestCaseTest(unittest.TestCase):
    """Tests for the parse_test_case function."""

    @parameterized.expand([
        # (inputs, output, expected_inputs, expected_output)

        # Integer inputs and output
        ("5", "10", (5,), 10),

        # Tuple input and integer output
        ("(1, 2)", "3", ((1, 2),), 3),

        # Tuple input and string output
        ("('a', 'b', 'c')", "'abc'", (("a", "b", "c"),), "abc"),

        # List input and list output
        ("[]", "[]", ([],), []),

        # Tuple input and output with mixed types
        ("(1, 'a', 3.5)", "(2, 'b', 4.5)", ((1, 'a', 3.5),), (2, 'b', 4.5)),
        (
            "[[1, 2], [3, 4]]",
            "[[5, 6], [7, 8]]",
            ([[1, 2], [3, 4]],),
            [[5, 6], [7, 8]]
        ),

        # Multiple inputs and single output
        (
            "['a', 2, 'c']\n[1, 2, 3]",
            "6",
            (['a', 2, 'c'], [1, 2, 3]),
            6
        ),
    ])
    def test_parse_functional_case(
        self, inputs: str, output: str, expected_inputs: tuple[Any, ...],
        expected_output: Any):
        """Parses a functional test case dictionary correctly."""
        data = {
            "testtype": "functional",
            "input": inputs,
            "output": output,
        }

        result = evaluate_codegen.parse_test_case(data)

        self.assertIsInstance(result, evaluate_codegen.FunctionalTestCase)
        self.assertEqual(result.inputs, expected_inputs)  # type: ignore
        self.assertEqual(result.expected_output, expected_output)  # type: ignore
    
    @parameterized.expand([
        # (inputs, output)

        # Invalid syntax in inputs
        ("5 +", "10"),

        # Invalid syntax in output
        ("5", "10 +"),

        # Non-literal in inputs
        ("open('file.txt')", "10"),
    ])
    def test_parse_functional_case_invalid_io_expressions(self, inputs: str, output: str):
        """Raises InvalidTestCaseExpressionException for invalid functional test case values."""
        data = {
            "input": inputs,
            "output": output,
            "testtype": "functional",
        }

    def test_parse_functional_case_invalid_output_raises(self):
        """Raises InvalidTestCaseOutputException for functional test case with invalid output."""
        data = {
            "input": "(1, 2)\n2",
            "output": "(3, 4)\n1",  # Invalid: multiple expressions
            "testtype": "functional",
        }

        with self.assertRaisesRegex(
            evaluate_codegen.InvalidTestCaseOutputException,
            "Functional test case output must be a single expression"):
            evaluate_codegen.parse_test_case(data)

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
        test_case = evaluate_codegen.StandardIOTestCase(
            stdin=stdin,
            expected_stdout=expected_stdout
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
        self.assertIn("Expected stdout: a, but got: 6", result.error_message)
    
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
