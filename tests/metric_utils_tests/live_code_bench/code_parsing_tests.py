import textwrap
import unittest

from eureka_ml_insights.metrics.live_code_bench import code_parsing


class FindFunctionPathTest(unittest.TestCase):
    """Tests for find_function_path function."""

    def test_find_function_at_module_level(self):
        """Test finding a function at module level."""
        src_code = "def my_function(): pass"
        result = code_parsing.find_function_path(src_code, "my_function")
        self.assertEqual(result, "my_function")

    def test_find_function_in_single_class(self):
        """Test finding a function inside a class."""
        src_code = textwrap.dedent("""
            class MyClass:
                def my_function(self):
                    pass
        """)
        result = code_parsing.find_function_path(src_code, "my_function")
        self.assertEqual(result, "MyClass.my_function")

    def test_find_function_in_nested_class(self):
        """Test finding a function in a nested class."""
        src_code = textwrap.dedent("""
            class OuterClass:
                class InnerClass:
                    def my_function(self):
                        pass
        """)
        result = code_parsing.find_function_path(src_code, "my_function")
        self.assertEqual(result, "OuterClass.InnerClass.my_function")

    def test_find_function_in_deeply_nested_class(self):
        """Test finding a function in deeply nested classes."""
        src_code = textwrap.dedent("""
            class Level1:
                class Level2:
                    class Level3:
                        def my_function(self):
                            pass
        """)
        result = code_parsing.find_function_path(src_code, "my_function")
        self.assertEqual(result, "Level1.Level2.Level3.my_function")

    def test_function_not_found(self):
        """Test behavior when function is not found."""
        src_code = "def another_function(): pass"
        with self.assertRaisesRegex(
            LookupError,
            "No function named 'my_function' found.",
        ):
            code_parsing.find_function_path(src_code, "my_function")
    
    def test_multiple_functions_found(self):
        """Test behavior when multiple functions with the same name are found."""
        src_code = textwrap.dedent("""
            class ClassA:
                def my_function(self):
                    pass

            class ClassB:
                def my_function(self):
                    pass
        """)
        with self.assertRaisesRegex(
            code_parsing.AmbiguousFunctionNameError,
            "Multiple functions named 'my_function' found: "
        ):
            code_parsing.find_function_path(src_code, "my_function")
    
    def test_invalid_syntax(self):
        """Test behavior when source code has invalid syntax."""
        src_code = "def my_function(: pass"  # Syntax error
        with self.assertRaises(SyntaxError):
            code_parsing.find_function_path(src_code, "my_function")


class IsPythonCodeValidTest(unittest.TestCase):
    """Tests for is_python_code_valid function."""

    def test_valid_code(self):
        """Test with valid Python code."""
        src_code = textwrap.dedent("""
            def my_function():
                return 42
        """)
        is_valid, error_message = code_parsing.is_python_code_valid(src_code)
        self.assertTrue(is_valid)
        self.assertEqual(error_message, "")

    def test_invalid_code(self):
        """Test with invalid Python code."""
        src_code = textwrap.dedent("""
            def my_function(
                return 42
        """)
        is_valid, error_message = code_parsing.is_python_code_valid(src_code)
        self.assertFalse(is_valid)
        self.assertIn("'(' was never closed", error_message)


if __name__ == "__main__":
    unittest.main()
