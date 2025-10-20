"""Defines some utilities for parsing text."""

import re


def extract_code(
        response: str | None, closing_think_token: str="</think>") -> str:
    """Extracts the code snippet from the model response.

    Only considers text after the last `closing_think_token`.

    Args:
        response: The model response as a string.
        closing_think_token: The token indicating the end of the model's thought
            process.

    Returns:
        The extracted code snippet as a string, or an empty string if no code
        snippet is found.
    """
    if not response:
        return ""
    
    if closing_think_token not in response:
        return ""

    # Remove content before and including the closing think token.
    content_after_thinking: str = (
        response.rpartition(closing_think_token)[2].strip())

    if not content_after_thinking:
        return ""

    # Try to find a code snippet in markdown format:
    #   ```python
    #   <code>
    #   ```
    # or
    #   ```
    #   <code>
    #   ```
    match = re.search(
        r"```(?:python)?\n(.*?)\n```", content_after_thinking, re.DOTALL)
    if match:
        return match.group(1).strip()

    return ""
