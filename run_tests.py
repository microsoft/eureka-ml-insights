import logging
import os
import sys
import unittest
from typing import List


def discover_all_tests_in_folder(test_folder: str) -> unittest.TestSuite:
    """
    Discovers all unit tests in the specified folder.

    Args:
        test_folder (str): Path to the folder containing test modules.

    Returns:
        unittest.TestSuite: The test suite containing all discovered tests.
    """
    loader = unittest.TestLoader()
    return loader.discover(test_folder, pattern="*_tests.py")


def discover_tests_in_suite_by_id(nested_suite: str, test_identifiers: List[str]) -> None:
    """
    Finds tests whose names include the any test_identifiers string.

    Args:
        nested_suite: unittest.TestSuite. The test suite to recursively find the desired tests in.
        test_identifiers: List of str. Any part of the names of the test class or test method to run.
                         e.g. "TestDataJoin" or "TestDataJoin.test_data_join"
    """
    # Find the tests that have any of the the test_identifiers in their name
    res_suite = unittest.TestSuite()

    def get_tests(test_cand, res_suite, test_identifiers):
        if isinstance(test_cand, unittest.suite.TestSuite):
            for sub_test in test_cand._tests:
                get_tests(sub_test, res_suite, test_identifiers)
        else:
            if any(test_identifier in str(test_cand) for test_identifier in test_identifiers):
                res_suite.addTest(test_cand)
        return res_suite

    return get_tests(nested_suite, res_suite, test_identifiers)


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR, format="%(filename)s - %(funcName)s - %(message)s")
    test_folder_path = os.path.join(os.getcwd(), "tests")

    # discover all tests in the test folder.
    suite = discover_all_tests_in_folder(test_folder_path)

    # if no arguments are provided, run all discovered tests.
    # Otherwise, filter the tests based on the test indentifiers in the commandline arguments.
    if len(sys.argv) >= 2:
        test_identifiers = sys.argv[1:]
        suite = discover_tests_in_suite_by_id(suite, test_identifiers)

    runner = unittest.TextTestRunner(verbosity=2)
    test_result: unittest.TestResult = runner.run(suite)

    if test_result.failures or test_result.errors:
        sys.exit(1)
