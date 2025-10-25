"""Defines the result of evaluating a test case."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class TestCaseResult:
    """The result of evaluating a test case.

    Attributes:
        success: Whether the test case passed.
        error_message: An optional error message if the test case failed.
    """
    passed: bool
    error_message: str = ""
