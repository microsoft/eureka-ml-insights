"""Parses functional test cases from the LiveCodeBench format."""

import ast
import json

from typing import Any

from eureka_ml_insights.metrics.live_code_bench import raw_test_case
from eureka_ml_insights.metrics.live_code_bench import functional_test_case


class InvalidTestCaseExpressionException(Exception):
    """Raised when a test case expression cannot be parsed."""


class InvalidTestCaseOutputException(Exception):
    """Raised when a test case output is invalid."""


def _convert_string_to_literal(expr: str) -> Any:
    """Converts a string expression to a Python literal.

    This function attempts to parse the input string as a JSON object first,
    and if that fails, it falls back to using `ast.literal_eval`.

    The use of JSON loads mirrors the original code in
    https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/evaluation/testing_util.py#L246C10-L246C26

    Args:
        expr: A string representation of a Python literal.

    Returns:
        The evaluated Python literal.

    Raises:
        InvalidTestCaseExpressionException: If the expression cannot be parsed.
    """
    expr = expr.strip()

    if not expr:
        raise InvalidTestCaseExpressionException(
            "Empty expression cannot be parsed.")

    try:
        return json.loads(expr)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(expr)
    except (ValueError, SyntaxError) as e:
        raise InvalidTestCaseExpressionException(
            f"Failed to parse expression: {expr!r}") from e


def _parse_functional_test_case_io(expr: str) -> tuple[Any, ...]:
    """Parses a functional test case input or output expression.

    Args:
        expr: A string representation of the input or output.
            Can be a single expression or multiple expressions
            separated by newlines (each representing an argument).

    Returns:
        A tuple of evaluated expressions.

    Raises:
        InvalidTestCaseExpressionException: If any expression cannot be parsed.
    """
    result: list[Any] = []
    for i, sub_expr in enumerate(expr.split("\n"), start=1):
        try:
            evaluated = _convert_string_to_literal(sub_expr)
            result.append(evaluated)
        except InvalidTestCaseExpressionException:
            raise InvalidTestCaseExpressionException(
                f"Failed to parse expression at line {i}: {sub_expr!r}")
    return tuple(result)


def parse_functional_test_case(
    test_case_dict: raw_test_case.RawTestCaseDict
) -> functional_test_case.FunctionalTestCase:
    """Parses a dictionary into a FunctionalTestCase.

    Args:
        test_case_dict: A dictionary with keys 'input' and 'output'.
            'input' should contain a string with the representation of the
            input as it would be passed to the function (e.g., a list or tuple).
            'output' should contain a string with the representation of the
            expected output. There should be only one expression for the output
            (i.e. a single line).
            Example:
                {
                    "input": "['a', 2, 'c']\n[1, 2, 3]",
                    "output": "6"
                }

    Returns:
        A FunctionalTestCase instance.

    Raises:
        InvalidTestCaseExpressionException: If the input or output expressions
            cannot be parsed as literals.
        InvalidTestCaseOutputException: If the output cannot be parsed into a
            single expression.
    """
    inputs = _parse_functional_test_case_io(test_case_dict["input"])
    output = _parse_functional_test_case_io(test_case_dict["output"])

    if len(output) != 1:
        # Function outputs should be a single expression. If the function
        # returns multiple values, they should be returned as a tuple.
        raise InvalidTestCaseOutputException(
            "Functional test case output must be a single expression. "
            f"Got {len(output)} expressions: {output}")

    output = output[0]

    return functional_test_case.FunctionalTestCase(inputs=inputs,
                                                   expected_output=output)
