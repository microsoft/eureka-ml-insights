import unittest

from typing import Any
from parameterized import parameterized

from eureka_ml_insights.metrics.live_code_bench import raw_test_case
from eureka_ml_insights.metrics.live_code_bench import functional_test_case
from eureka_ml_insights.metrics.live_code_bench import functional_test_case_parser


class FunctionalTestCaseParserTests(unittest.TestCase):
    """Tests for the parse_functional_test_case function."""

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

        # JSON literals booleans
        (
            "(1, 2, 3)",
            "[true, false, null]",
            ((1, 2, 3),),
            [True, False, None]
        ),

        # Python literals booleans
        (
            "(1, 2, 3)",
            "[True, False, None]",
            ((1, 2, 3),),
            [True, False, None]
        ),
    ])
    def test_parse_functional_case(
        self, inputs: str, output: str,
        expected_inputs: tuple[Any, ...],
        expected_output: Any):
        """Parses a functional test case dictionary correctly."""
        data: raw_test_case.RawTestCaseDict = {
            "testtype": "functional",
            "input": inputs,
            "output": output,
        }

        result = functional_test_case_parser.parse_functional_test_case(data)

        self.assertIsInstance(result, functional_test_case.FunctionalTestCase)
        self.assertEqual(result.inputs, expected_inputs)  # type: ignore
        self.assertEqual(result.expected_output, expected_output)  # type: ignore
    
    @parameterized.expand([
        # (inputs, output)

        # Invalid syntax in inputs
        ("5 +", "10"),

        # Invalid syntax in output
        ("(1, 2)", "3 +"),

        # Non-literal in inputs
        ("open('file.txt')", "10"),

        # Non-literal in output
        ("5", "open('file.txt')"),

        # Empty input expression
        ("'a'\n", "10"),
    ])
    def test_parse_functional_case_invalid_input(self, inputs: str, output: str):
        """Raises InvalidTestCaseExpressionException for invalid functional test case values."""
        data: raw_test_case.RawTestCaseDict = {
            "input": inputs,
            "output": output,
            "testtype": "functional",
        }

        with self.assertRaises(
            functional_test_case_parser.InvalidTestCaseExpressionException):
            functional_test_case_parser.parse_functional_test_case(data)

    def test_parse_functional_case_invalid_output_raises(self):
        """Raises InvalidTestCaseOutputException for functional test case with invalid output."""
        data: raw_test_case.RawTestCaseDict = {
            "input": "(1, 2)\n2",
            "output": "(3, 4)\n1",  # Invalid: multiple expressions
            "testtype": "functional",
        }

        with self.assertRaises(
            functional_test_case_parser.InvalidTestCaseOutputException):
            functional_test_case_parser.parse_functional_test_case(data)
