"""Defines the raw test case dictionary structure for LiveCodeBench test cases."""

from typing import TypedDict


class RawTestCaseDict(TypedDict):
    """A raw test case dictionary as obtained from the LiveCodeBench test cases.

    Attributes:
        input: The input string for the test case.
        output: The expected output string for the test case.
        testtype: The type of the test case.
    """
    input: str
    output: str
    testtype: str
