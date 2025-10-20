"""Unit tests for the functions in parsing.py.

To run:
    $ python -m unittest \
        eureka_ml_insights.data_utils.live_code_bench.parsing_test
"""

import unittest
from parameterized import parameterized

from eureka_ml_insights.data_utils.live_code_bench import parsing


class ExtractCodeBlocksTest(unittest.TestCase):
    """Tests for extract_code_blocks."""

    @parameterized.expand([
        # (response, closing_think_token, expected_blocks)
        (None, "", []),
        ("", "", []),

        # No closing think token, think token required → ignored
        ("Here is some code:\n```python\nprint('Hello')\n```", "</think>", []),

        # No closing think token, not required → extracts code
        (
            "Here is some code:\n```python\nprint('Hello')\n```",
            "",
            ["print('Hello')"]
        ),

        # Nothing after closing think token
        (
            "Some thought process.</think>",
            "</think>",
            []
        ),

        # No code after closing think token
        (
            "Thoughts here.</think>\nNo code snippet.",
            "</think>",
            []
        ),

        # Code block before closing think token (should be ignored if token
        # required)
        (
            "```python\nprint('Hello, World!')\n```\nSome thoughts.</think>",
            "</think>",
            []
        ),

        # Code block before closing think token (should be included since
        # token not required)
        (
            "```python\nprint('Hello, World!')\n```\nSome thoughts.</think>",
            "",
            ["print('Hello, World!')"]
        ),

        # Malformed (missing closing ``` after think)
        (
            "Thoughts here.</think>\n```python\nprint('Hello, World!')",
            "</think>",
            []
        ),

        # Proper code block after closing think token - python specified
        (
            "Thoughts here.</think>\n```python\nprint('Hello, World!')\n```",
            "</think>",
            ["print('Hello, World!')"]
        ),

        # Proper code block after closing think token - no language tag
        (
            "Thoughts here.</think>\n```\nprint('Hello, World!')\n```",
            "</think>",
            ["print('Hello, World!')"]
        ),

        # Multiple code blocks after closing think token
        (
            "</think>\n```python\nx=1\n```\n```python\ny=2\n```",
            "</think>",
            ["x=1", "y=2"]
        ),
    ])
    def test_extract_code_blocks(
        self, response: str | None, closing_think_token: str,
        expected_blocks: list[str]):
        extracted_blocks = parsing.extract_code_blocks(
            response, closing_think_token)
        self.assertEqual(extracted_blocks, expected_blocks)


class ExtractLastCodeBlockTest(unittest.TestCase):
    """Tests for extract_last_code_block."""

    @parameterized.expand([
        # (response, closing_think_token, expected_code)
        (None, "", ""),
        ("", "", ""),

        # No closing think token, think token required → ignored
        ("Here is some code:\n```python\nprint('Hello')\n```", "</think>", ""),

        # No closing think token, not required → extracts code
        (
            "Here is some code:\n```python\nprint('Hello')\n```",
            "",
            "print('Hello')"
        ),

        # Nothing after closing think token
        ("Some thought process.</think>", "</think>", ""),

        # No code after closing think token
        ("Thoughts here.</think>\nNo code snippet.", "</think>", ""),

        # Code block before closing think token (should be ignored since
        # token required)
        (
            "```python\nprint('Hello, World!')\n```\nSome thoughts.</think>",
            "</think>",
            ""
        ),

        # Code block before closing think token (should be included since
        # token not required)
        (
            "```python\nprint('Hello, World!')\n```\nSome thoughts.</think>",
            "",
            "print('Hello, World!')"
        ),

        # Multiple code blocks → only last one should be returned
        (
            "</think>\n```python\nx=1\n```\n```python\ny=2\n```",
            "</think>",
            "y=2"
        ),
    ])
    def test_extract_last_code_block(self, response, closing_think_token, expected_code):
        extracted_code = parsing.extract_last_code_block(response, closing_think_token)
        self.assertEqual(extracted_code, expected_code)