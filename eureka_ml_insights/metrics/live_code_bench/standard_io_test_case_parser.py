"""Parses standard I/O test cases from the LiveCodeBench format."""

from eureka_ml_insights.metrics.live_code_bench import raw_test_case
from eureka_ml_insights.metrics.live_code_bench import standard_io_test_case


def parse_standard_io_test_case(
    test_case_dict: raw_test_case.RawTestCaseDict
) -> standard_io_test_case.StandardIOTestCase:
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
    return standard_io_test_case.StandardIOTestCase(
        stdin=stdin, expected_stdout=expected_stdout)
