import subprocess
import unittest

from eureka_ml_insights.core.job_runner.jobs import python_script_job


class TestPythonScriptJob(unittest.TestCase):
    """Unit tests for PythonScriptJob."""

    def test_run_basic_script(self):
        script = "print('hello world')"
        job = python_script_job.PythonScriptJob(script=script)

        result = subprocess.run(job.get_command(), capture_output=True)
        job_result = job.deserialize_result(result.stdout, result.stderr,
                                            result.returncode)

        self.assertEqual(job_result.stdout_str.strip(), "hello world")
        self.assertEqual(job_result.stderr_str, "")
        self.assertEqual(job_result.returncode, 0)

    def test_run_script_with_error(self):
        script = "import sys\nsys.exit(5)"
        job = python_script_job.PythonScriptJob(script=script)

        result = subprocess.run(job.get_command(), capture_output=True)
        job_result = job.deserialize_result(result.stdout, result.stderr,
                                            result.returncode)

        self.assertEqual(job_result.returncode, 5)
        self.assertEqual(job_result.stdout_str, "")
        self.assertEqual(job_result.stderr_str, "")

    def test_run_script_with_stdin(self):
        script = "import sys\nprint(sys.stdin.read())"
        job = python_script_job.PythonScriptJob(script=script,
                                                stdin="input data")

        result = subprocess.run(job.get_command(),
                                input=job.serialize_input(),
                                capture_output=True)
        job_result = job.deserialize_result(result.stdout, result.stderr,
                                            result.returncode)

        self.assertEqual(job_result.stdout_str.strip(), "input data")
        self.assertEqual(job_result.stderr_str, "")
        self.assertEqual(job_result.returncode, 0)

    def test_run_script_with_bytes_stdin(self):
        script = "import sys\nprint(sys.stdin.read())"
        job = python_script_job.PythonScriptJob(script=script,
                                                stdin=b"bytes data")

        result = subprocess.run(job.get_command(),
                                input=job.serialize_input(),
                                capture_output=True)
        job_result = job.deserialize_result(result.stdout, result.stderr,
                                            result.returncode)

        self.assertEqual(job_result.stdout_str.strip(), "bytes data")
        self.assertEqual(job_result.stderr_str, "")
        self.assertEqual(job_result.returncode, 0)

    def test_deserialize_result_properties(self):
        job_result = python_script_job.PythonScriptJobResult(stdout=b"out",
                                                             stderr=b"err",
                                                             returncode=42)
        self.assertEqual(job_result.stdout_str, "out")
        self.assertEqual(job_result.stderr_str, "err")
        self.assertEqual(job_result.returncode, 42)


if __name__ == "__main__":
    unittest.main()
