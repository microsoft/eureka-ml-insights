"""Defines utilities to run LiveCodeBench code generation test cases."""

import ast
import dataclasses
import datetime

from typing import Any, TypedDict, cast

from eureka_ml_insights.metrics.live_code_bench import code_execution


class RawTestCaseDict(TypedDict):
    """A raw test case dictionary as obtained from the LiveCodeBench test cases.

    Attributes:
        testtype: The type of the test case.
        input: The input string for the test case.
        output: The expected output string for the test case.
    """
    input: str
    output: str
    testtype: str


@dataclasses.dataclass(frozen=True)
class FunctionalTestCase:
    """A test case for evaluating functions.

    Attributes:
        inputs: The inputs to the function as a tuple of arguments. Will be
            unpacked when calling the function: func(*inputs).
        expected_output: The expected output from the function.
    """
    inputs: tuple[Any, ...]
    expected_output: Any


@dataclasses.dataclass(frozen=True)
class StandardIOTestCase:
    """A test case for evaluating scripts that read from stdin and write to stdout.

    Attributes:
        stdin: The input string to be provided to the script via stdin.
        expected_stdout: The expected output string from the script via stdout.
    """
    stdin: str
    expected_stdout: str


@dataclasses.dataclass(frozen=True)
class TestCaseResult:
    """The result of evaluating a test case.

    Attributes:
        success: Whether the test case passed.
        error_message: An optional error message if the test case failed.
    """
    passed: bool
    error_message: str = ""


class InvalidTestCaseExpressionException(Exception):
    """Raised when a test case expression cannot be parsed."""


class InvalidTestCaseOutputException(Exception):
    """Raised when a test case output is invalid."""


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
        sub_expr = sub_expr.strip()
        if not sub_expr:
            continue
        try:
            evaluated = ast.literal_eval(sub_expr)
        except (ValueError, SyntaxError) as e:
            raise InvalidTestCaseExpressionException(
                f"Failed to parse expression on line {i}: {sub_expr!r}") from e
        result.append(evaluated)
    return tuple(result)


def _parse_functional_test_case(
        test_case_dict: RawTestCaseDict) -> FunctionalTestCase:
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

    return FunctionalTestCase(
        inputs=inputs,
        expected_output=output
    )


def _parse_standard_io_test_case(
        test_case_dict: RawTestCaseDict) -> StandardIOTestCase:
    """Parses a dictionary into a StandardIOTestCase.

    Args:
        test_case_dict: A dictionary with keys 'input' and 'output'.
            'input' should contain the input string to be provided to the
                script.
            'output' should contain the expected output string from the script.
            Example:
                {
                    "input": "input data",
                    "output": "expected output"
                }

    Returns:
        A StandardIOTestCase instance.
    """
    stdin = test_case_dict["input"]
    expected_stdout = test_case_dict["output"]
    return StandardIOTestCase(stdin=stdin, expected_stdout=expected_stdout)


def parse_test_case(
        test_case_dict: dict[str, str]
) -> FunctionalTestCase | StandardIOTestCase:
    """Parses a test case dictionary into the appropriate test case dataclass.

    Args:
        test_case_dict: A dictionary representing the test case.
        It must contain 'testtype', 'input', and 'output' keys.
        'testtype' should be either 'functional' or 'stdin'.

    Returns:
        A FunctionalTestCase or StandardIOTestCase instance.
    
    Raises:
        ValueError: If the test case type is unknown.
    """
    required_keys = {"testtype", "input", "output"}
    missing = required_keys - test_case_dict.keys()
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(missing)}")

    validated_test_case_dict = cast(RawTestCaseDict, test_case_dict)

    match validated_test_case_dict["testtype"]:
        case "functional":
            return _parse_functional_test_case(validated_test_case_dict)
        case "stdin":
            return _parse_standard_io_test_case(validated_test_case_dict)
        case _:
            raise ValueError(
                f"Unknown test type: {validated_test_case_dict['testtype']}")


def _evaluate_functional_test_case(
        src_code: str,
        function_name: str,
        test_case: FunctionalTestCase,
        timeout: datetime.timedelta | None = None) -> TestCaseResult:
    """Evaluates a functional test case against the provided source code.

    Args:
        src_code: The source code containing the function definition.
        function_name: The name of the function to be tested. If the function
            is part of a class, provide the full path (e.g.,
            'MyClass.my_function').
        test_case: The FunctionalTestCase instance.
        timeout: An optional timeout for the code execution.

    Returns:
        A TestCaseResult instance indicating whether the test case passed.
    """
    job = code_execution.FunctionJob(src_code=src_code,
                                     function_name=function_name,
                                     args=test_case.inputs,
                                     timeout=timeout)

    result: code_execution.FunctionResult = (
        code_execution.execute_function(job))
    
    if result.success and result.return_value == test_case.expected_output:
        return TestCaseResult(passed=True)
    elif not result.success:
        return TestCaseResult(passed=False,
                              error_message=result.error_message)
    else:
        return TestCaseResult(
            passed=False,
            error_message=(
                f"Expected output: {test_case.expected_output}, "
                f"but got: {result.return_value}"))


def _evaluate_standard_io_test_case(
        src_code: str,
        test_case: StandardIOTestCase,
        timeout: datetime.timedelta | None = None) -> TestCaseResult:
    """Evaluates a standard I/O test case against the provided source code.

    The source code is expected to be a script that reads from stdin and
    writes the answer to stdout.

    Args:
        src_code: The source code to be tested.
        test_case: The StandardIOTestCase instance.
        timeout: An optional timeout for the code execution.
    
    Returns:
        A TestCaseResult instance indicating whether the test case passed.
    """
    job = code_execution.ScriptJob(script=src_code,
                                   stdin_input=test_case.stdin,
                                   timeout=timeout)

    result: code_execution.ScriptResult = (code_execution.execute_script(job))

    # Strip trailing newlines for comparison.
    # The generated code may use sys.stdout.write which does not add a newline.
    received_stdout: str = result.stdout.rstrip("\n")
    expected_stdout: str = test_case.expected_stdout.rstrip("\n")

    if result.success and received_stdout == expected_stdout:
        return TestCaseResult(passed=True)
    elif not result.success:
        return TestCaseResult(passed=False,
                              error_message=result.error_message)
    else:
        return TestCaseResult(
            passed=False,
            error_message=(
                f"Expected stdout: {expected_stdout}, "
                f"but got: {received_stdout}"))


def evaluate_test_case(
        src_code: str,
        test_case: FunctionalTestCase | StandardIOTestCase,
        function_name: str = "",
        timeout: datetime.timedelta | None = None) -> TestCaseResult:
    """Evaluates a test case against the provided source code.

    Args:
        src_code: The source code to be tested.
        test_case: The test case to evaluate.
        function_name: The name of the function to be tested. This is only
            required for FunctionalTestCase instances. If the function is part
            of a class, provide the full path (e.g., 'MyClass.my_function').
        timeout: An optional timeout for the code execution.

    Returns:
        A TestCaseResult instance indicating whether the test case passed.
    
    Raises:
        ValueError: If the test case type is unknown or if function_name is not
            provided for FunctionalTestCase.
    """
    if isinstance(test_case, FunctionalTestCase):
        if not function_name:
            raise ValueError(
                "function_name must be provided for FunctionalTestCase.")
        return _evaluate_functional_test_case(
            src_code=src_code,
            function_name=function_name,
            test_case=test_case,
            timeout=timeout
        )
    elif isinstance(test_case, StandardIOTestCase):
        return _evaluate_standard_io_test_case(
            src_code=src_code,
            test_case=test_case,
            timeout=timeout
        )
    else:
        raise ValueError(f"Unknown test case type: {type(test_case)}")
