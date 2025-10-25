"""Defines the structure of a functional test case.

These test cases consist of Python primitive inputs and expected outputs.
"""

import dataclasses

from typing import Any


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
