"""Defines some utilities for parsing text."""

import re


def extract_code_blocks(
        response: str | None, closing_think_token: str = "") -> list[str]:
    """Extracts all code snippets from a model response.

    Only considers text after the last `closing_think_token` if provided.

    Args:
        response: The model response as a string.
        closing_think_token: The token marking the end of the model's thought
            process (e.g., "</think>").

    Returns:
        A list of all extracted code snippets (possibly empty).
    """
    if not response:
        return []
    
    if closing_think_token and closing_think_token not in response:
        return []

    # Restrict to text after the last think token, if present
    response_to_consider = (
        response.rpartition(closing_think_token)[2].strip()
        if closing_think_token
        else response.strip()
    )

    if not response_to_consider:
        return []

    # Find all markdown-style code blocks, optionally with a language tag
    # Matches both ```python\n<code>\n``` and ```\n<code>\n```
    matches = re.findall(
        r"```(?:python)?\n(.*?)\n```", response_to_consider, re.DOTALL)

    # Strip whitespace around each code snippet
    return [m.strip() for m in matches]


def extract_last_code_block(
        response: str | None, closing_think_token: str = "") -> str:
    """Extracts the last code snippet from a model response.

    Args:
        response: The model response as a string.
        closing_think_token: The token marking the end of the model's thought
            process (e.g., "</think>").

    Returns:
        The last extracted code snippet, or an empty string if none found.
    """
    blocks = extract_code_blocks(response, closing_think_token)
    return blocks[-1] if blocks else ""
