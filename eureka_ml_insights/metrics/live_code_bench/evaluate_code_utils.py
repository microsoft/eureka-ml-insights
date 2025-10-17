import datetime
import dataclasses
import json
import ast

from typing import Any
from collections.abc import Callable
from concurrent import futures


@dataclasses.dataclass(frozen=True)
class TestCase:
    """Represents a single test case for evaluating generated code.

    Attributes:
        inputs: A sequence containing inputs to the function. These should
            be the actual data types as they would be passed to the function.
            E.g., if the function takes a list of integers and a string, 
            the inputs would be [[1, 2, 3], 'hello'].
        expected_output: The expected output from the function as an actual
            Python object (not as a string). For example, [1, 2, 3] or 'hello'.
    """
    inputs: list[Any]
    expected_output: Any


@dataclasses.dataclass(frozen=True)
class TestResult:
    """Represents the result of evaluating a single test case.

    Attributes:
        passed: Whether the test case passed (i.e., the function's output
            matched the expected output).
        error_message: If an error occurred during execution, this contains
            the error message. Otherwise, it is an empty string.
    """
    passed: bool
    error_message: str = ""


def parse_test_cases(test_cases_json: str) -> list[TestCase]:
    """Parses a JSON string of test cases into a list of TestCase objects.
    
    Args:
        test_cases_json: A JSON string representing a list of test cases.
            Each test case should be a dictionary with:
            - 'inputs': A string containing arguments (one per line) as Python
               literals. E.g., "[1, 2, 3]\n'hello'"
            - 'expected_output': A string containing the expected output as a
               Python literal. E.g., "[1, 2, 3]" or "'hello'".
    
    Returns:
        A list of TestCase objects with parsed Python objects.
        
    Raises:
        ValueError: If JSON is invalid or test cases are malformed.
    """
    try:
        test_cases_data = json.loads(test_cases_json)
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON format for test cases.") from e

    if not isinstance(test_cases_data, list):
        raise ValueError(
            "Test cases JSON should represent a list of test cases.")

    parsed_cases: list[TestCase] = []
    for i, case in enumerate(test_cases_data):
        try:
            # Parse inputs: split by newlines and evaluate each as Python literal
            inputs_str = case["inputs"]
            input_lines = [
                line.strip() for line in inputs_str.split("\n")
                if line.strip()
            ]
            inputs = [ast.literal_eval(line) for line in input_lines]

            # Parse expected output
            expected_output = ast.literal_eval(case["expected_output"])

            parsed_cases.append(
                TestCase(inputs=inputs, expected_output=expected_output))

        except (KeyError, ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid format in test case {i}: {e}") from e

    return parsed_cases


def _normalize_for_comparison(val: Any) -> Any:
    """Normalize sequences to tuples for comparison.

    Args:
        val: The value to normalize.
    
    Returns:
        The normalized value. Lists and sets are converted to tuples. Other
        types are returned unchanged.
    """
    if isinstance(val, (list, set)):
        return tuple(val)
    return val


def evaluate_function(function: Callable[..., Any],
                      test_case: TestCase,
                      timeout: datetime.timedelta | None = None) -> TestResult:
    """Evaluates a function against a single test case.
    
    Args:
        function: The function to be tested.
        test_case: A TestCase object.
        timeout: Maximum allowed time for function execution. If the function
            exceeds this time, it is considered a failure.
    
    Returns:
        A TestResult indicating whether the test passed, any error message,
        and error code.
    """
    timeout_seconds: float | None = (timeout.total_seconds()
                                     if timeout is not None else None)
    try:
        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(function, *test_case.inputs)
            result = future.result(timeout=timeout_seconds)

    except futures.TimeoutError:
        return TestResult(
            passed=False,
            error_message=("Timeout: Function execution exceeded "
                           f"{timeout_seconds} seconds"))

    except Exception as e:
        return TestResult(
            passed=False,
            error_message=f"Exception during execution: {str(e)}")

    result_normalized = _normalize_for_comparison(result)
    expected_normalized = _normalize_for_comparison(test_case.expected_output)

    passed = (result_normalized == expected_normalized)

    if passed:
        return TestResult(passed=True)
    else:
        return TestResult(
            passed=False,
            error_message=(
                f"Wrong output: Expected {test_case.expected_output!r}, "
                f"got {result!r}"))


def _is_main_str(node: ast.AST) -> bool:
    """Check if node is a string literal with value '__main__'."""
    return (isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value == "__main__")


def _is_name_var(node: ast.AST) -> bool:
    """Check if node is a Name node with id '__name__'."""
    return isinstance(node, ast.Name) and node.id == "__name__"


def clean_if_main_block(code: str) -> str:
    """Remove `if __name__ == '__main__'` or equivalent block from code.

    This safely detects the main guard regardless of quote style,
    comparison order, or parentheses.

    Args:
        code: The code string to clean.

    Returns:
        The cleaned code string without the `if __name__ == '__main__'` block.

    Raises:
        ValueError: If the code cannot be parsed.
    """
    try:
        tree = ast.parse(code)
    except Exception as e:
        raise ValueError("Failed to parse code for cleaning.") from e

    new_body = []

    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            if (isinstance(test, ast.Compare)
                    and isinstance(test.ops[0], ast.Eq)
                    and len(test.comparators) == 1):
                left, right = test.left, test.comparators[0]

                # Allow both `__name__ == "__main__"`
                # and `"__main__" == __name__`
                if ((_is_name_var(left) and _is_main_str(right))
                        or (_is_main_str(left) and _is_name_var(right))):
                    new_body.extend(node.body)
                    continue

        new_body.append(node)

    tree.body = new_body
    return ast.unparse(tree)
