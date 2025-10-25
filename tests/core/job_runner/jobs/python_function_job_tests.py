import pickle
import unittest

from eureka_ml_insights.core.job_runner.jobs import python_function_job


class TestPythonFunctionJob(unittest.TestCase):

    def test_serialize_input(self):
        job = python_function_job.PythonFunctionJob(
            src_script="def foo(): return 123", function_name="foo")
        serialized = job.serialize_input()
        self.assertIsInstance(serialized, bytes)

    def test_command_generation(self):
        job = python_function_job.PythonFunctionJob(
            src_script="", function_name="")
        command = job.get_command()
        self.assertIsInstance(command, list)

    def test_deserialize_result_success(self):
        result_dict = {
            "result": 42,
            "stdout": "Hello\n",
            "stderr": "",
            "exception_class_name": None,
            "exception_msg": None,
        }
        stdout = pickle.dumps(result_dict)

        job = python_function_job.PythonFunctionJob(
            "", "")
        result = job.deserialize_result(stdout, b"", 0)
        self.assertIsInstance(
            result,
            python_function_job.PythonFunctionJobResult)
        self.assertTrue(result.success)
        self.assertEqual(result.return_value, 42)
        self.assertEqual(result.stdout, "Hello\n")
        self.assertEqual(result.stderr, "")

    def test_deserialize_result_failure(self):
        result_dict = {
            "result": None,
            "stdout": "",
            "stderr": "error",
            "exception_class_name": "ValueError",
            "exception_msg": "Something went wrong",
        }
        stdout = pickle.dumps(result_dict)
        job = python_function_job.PythonFunctionJob(
            "", "")
        result = job.deserialize_result(stdout, b"", 1)
        self.assertFalse(result.success)
        self.assertEqual(result.exception_class_name, "ValueError")
        self.assertEqual(result.exception_msg, "Something went wrong")

    def test_deserialize_invalid_pickle(self):
        job = python_function_job.PythonFunctionJob(
            "", "")
        invalid_bytes = b"not a pickle"
        with self.assertRaises(ValueError) as cm:
            job.deserialize_result(invalid_bytes, b"", 0)
        self.assertIn("Failed to deserialize job result", str(cm.exception))

    def test_deserialize_missing_keys(self):
        job = python_function_job.PythonFunctionJob(
            "", "")
        incomplete_result = pickle.dumps({"result": 1})
        with self.assertRaises(ValueError) as cm:
            job.deserialize_result(incomplete_result, b"", 0)
        self.assertIn("Missing keys in result", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
