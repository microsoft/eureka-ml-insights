"""Defines utilities for parsing Python code."""

import ast


class AmbiguousFunctionNameError(Exception):
    """Raised when multiple functions with the same name are found in the code."""
    pass


class _FunctionFinder(ast.NodeVisitor):
    """AST visitor that finds function definitions by name.

    Say the code is as follows:
        class MyClass:
            def my_function(self):
                pass
    
    If we search for "my_function", this visitor will record the full path
    "MyClass.my_function" in its matches.

    Attributes:
        matches: A list of full function names that match the searched name.
    """
    def __init__(self, func_name: str):
        """Initializes the FunctionFinder.

        Args:
            func_name: The name of the function to search for.
        """
        self.matches: list[str] = []
        self._class_stack: list[str] = []
        self._func_name = func_name

    def visit_ClassDef(self, node):
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node):
        if node.name == self._func_name:
            if self._class_stack:
                full_name = ".".join(self._class_stack + [node.name])
            else:
                full_name = node.name
            self.matches.append(full_name)
        self.generic_visit(node)


def find_function_path(src_code: str, func_name: str) -> str:
    """Finds the full path of a function in the given source code.

    Example:
        find_function_path("def my_function(): pass", "my_function")
        Returns: "my_function"

        find_function_path(
            "class MyClass:\n    def my_function(self): pass",
            "my_function"
        )
        Returns: "MyClass.my_function"

    Args:
        src_code: The source code to search.
        func_name: The name of the function to find.

    Returns:
        The full path of the function if found, else an empty string.
    
    Raises:
        LookupError: If no function with the given name is found.
        AmbiguousFunctionNameError: If multiple functions with the same name are
            found.
    """
    tree = ast.parse(src_code)
    finder = _FunctionFinder(func_name)
    finder.visit(tree)

    if not finder.matches:
        raise LookupError(f"No function named '{func_name}' found.")
    if len(finder.matches) > 1:
        raise AmbiguousFunctionNameError(
            f"Multiple functions named '{func_name}' found: {finder.matches}")
    return finder.matches[0]


def is_python_code_valid(src_code: str) -> tuple[bool, str]:
    """Checks if the given Python source code is syntactically valid.

    Args:
        src_code: The source code to check.
    
    Returns:
        A tuple (is_valid, error_message). is_valid is True if the code is
        valid, False otherwise. error_message contains the syntax error message
        if the code is invalid, or an empty string if the code is valid.
    """
    try:
        ast.parse(src_code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)
