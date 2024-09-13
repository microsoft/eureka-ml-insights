# write unit tests for the classes in data_utils/transform.py

import logging
import unittest

import pandas as pd

from eureka_ml_insights.data_utils.dna_utils import DNAParseLabel

log = logging.getLogger("DNA_Data_Transform_tests")


class TestDataTransform(unittest.TestCase):
    def setUp(self):
        testcases = ["1", "2", "0", "6", "5", "-1", "t", "10", ""]
        self.df = pd.DataFrame({"A": ["output <answer>{0}</answer>".format(s) for s in testcases]})

    def test_parse_label_updated_logic(self):
        log.info("Testing DNA Parsing with updated parser")
        transform = DNAParseLabel("A", "B", "C", use_updated_metric=True)
        result = transform.transform(self.df)
        self.assertListEqual(list(result.columns), ["A", "B", "C"])
        self.assertListEqual(list(result["B"]), [1, 2, 0, 6, 5, -1, -1, -1, -1])
        self.assertListEqual(list(result["C"]), [1, 1, 1, 0, 1, 0, 0, 0, 0])

    def test_parse_label(self):
        log.info("Testing DNA Parsing with parser from original benchmark logic")
        transform = DNAParseLabel("A", "B", "C", use_updated_metric=False)
        result = transform.transform(self.df)
        self.assertListEqual(list(result.columns), ["A", "B", "C"])
        self.assertListEqual(list(result["B"]), [1, 2, 0, 6, 5, -1, -1, 1, -1])
        self.assertListEqual(list(result["C"]), [1, 1, 1, 0, 1, 1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
