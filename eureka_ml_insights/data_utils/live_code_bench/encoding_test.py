"""Unit tests for the functions in encoding.py.

To run:
    $ python -m unittest \
        eureka_ml_insights.data_utils.live_code_bench.encoding_test
"""

import unittest

from parameterized import parameterized
from typing import Any

from eureka_ml_insights.data_utils.live_code_bench import encoding


class EncodeDecodeTestCasesTest(unittest.TestCase):
    """Tests for encode_test_cases() and decode_test_cases()."""

    @parameterized.expand([
        (
            # None input returns None
            None,
            False,
            None,
        ),
        (
            # Plain JSON encode/decode
            {"test_case_1": "value1", "test_case_2": "value2"},
            False,
            {"test_case_1": "value1", "test_case_2": "value2"},
        ),
        (
            # Compressed encode/decode
            {"example_key": "example_value"},
            True,
            {"example_key": "example_value"},
        ),
    ])
    def test_successful_encode_decode_round_trip(
        self, test_cases: Any, compress: bool, expected_output: Any):
        """Ensure encode() + decode() round-trips successfully."""
        encoded = encoding.encode_test_cases(test_cases, compress=compress)
        decoded = encoding.decode_test_cases(encoded)
        self.assertEqual(decoded, expected_output)

    def test_decode_invalid_input_raises(self):
        """Invalid inputs should raise a ValueError with descriptive message."""
        invalid_input = "This is not valid encoded test cases."
        with self.assertRaisesRegex(
            ValueError, "Failed to decode test cases"):
            encoding.decode_test_cases(invalid_input)

    def test_encode_invalid_data_raises(self):
        """Unserializable objects should raise a ValueError during encoding."""
        unserializable = {"key": set([1, 2, 3])}
        with self.assertRaisesRegex(
            ValueError, "Failed to encode test cases"):
            encoding.encode_test_cases(unserializable)