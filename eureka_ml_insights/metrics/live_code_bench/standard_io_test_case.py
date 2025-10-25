"""Defines the structure of a standard I/O test case.

These test cases consist of input strings provided via stdin and expected
output strings read from stdout.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class StandardIOTestCase:
    """A test case for evaluating scripts that read from stdin and write to stdout.

    Attributes:
        stdin: The input string to be provided to the script via stdin.
        expected_stdout: The expected output string from the script via stdout.
    """
    stdin: str
    expected_stdout: str
