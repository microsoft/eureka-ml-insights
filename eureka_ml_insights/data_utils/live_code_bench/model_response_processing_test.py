"""Unit tests for the functions in model_response_processing.py.

To run:
    $ python -m unittest \
        eureka_ml_insights.data_utils.live_code_bench.model_response_processing_test
"""

import unittest

from parameterized import parameterized

from eureka_ml_insights.data_utils.live_code_bench import model_response_processing


class ExtractCodeTest(unittest.TestCase):

    @parameterized.expand([
        (
            None,
            "",
        ),
        (
            "",
            "",
        ),
        (
            # No closing think token
            "Here is some code:\n```python\nprint('Hello')\n```",
            "",
        ),
        (
            # Nothing after closing think token
            "Some thought process.</think>",
            "",
        ),
        (
            # No code after think token
            "Thoughts here.</think>\nNo code snippet.",
            "",
        ),
        (
            # Code block before closing think token
            "```python\nprint('Hello, World!')\n```\nSome thoughts.</think>",
            "",
        ),
        (
            # Malformed code block after closing think token
            "Thoughts here.</think>\n```python\nprint('Hello, World!')",
            "",
        ),
        (
            # Code block after closing think token - python specified
            "Thoughts here.</think>\n```python\nprint('Hello, World!')\n```",
            "print('Hello, World!')",
        ),
        (
            # Code block after closing think token - no language specified
            "Thoughts here.</think>\n```\nprint('Hello, World!')\n```",
            "print('Hello, World!')",
        ),
    ])
    def test_extract_code(
            self,
            response: str | None,
            expected_code: str):
        extracted_code = model_response_processing.extract_code(response)
        self.assertEqual(extracted_code, expected_code)