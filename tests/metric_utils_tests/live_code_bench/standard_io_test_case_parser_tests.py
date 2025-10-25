import unittest

from eureka_ml_insights.metrics.live_code_bench import raw_test_case
from eureka_ml_insights.metrics.live_code_bench import standard_io_test_case
from eureka_ml_insights.metrics.live_code_bench import standard_io_test_case_parser

class StandardIOTestCaseParserTests(unittest.TestCase):
    def test_parse_stdio_test_case(self):
        """Parses a standard I/O test case dictionary correctly."""
        data: raw_test_case.RawTestCaseDict = {
            "testtype": "stdin",
            "input": "5\n6",
            "output": "6\n",
        }

        result = standard_io_test_case_parser.parse_standard_io_test_case(data)

        self.assertIsInstance(result, standard_io_test_case.StandardIOTestCase)
        self.assertEqual(result.stdin, "5\n6")  # type: ignore
        self.assertEqual(result.expected_stdout, "6\n")  # type: ignore
