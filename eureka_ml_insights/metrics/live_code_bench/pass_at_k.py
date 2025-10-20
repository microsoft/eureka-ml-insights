"""Defines functions to compute Pass@K metric."""
import math

from collections.abc import Sequence


def estimate_pass_at_k(
    num_attempts: int, num_correct: int, k: int
) -> float:
    """Computes the Pass@K metric for the given test results.

    Args:
        num_attempts: An integer representing the total number of attempts.
        num_correct: An integer representing the number of correct attempts.
        k: An integer representing the K value for the Pass@K metric.

    Returns:
        A float representing the Pass@K metric.
    
    Raises:
        ValueError: If K is not a positive integer or if num_correct exceeds
            num_attempts.
    """
    if num_correct > num_attempts:
        raise ValueError(
            "Number of correct attempts cannot exceed total attempts. "
            f"Got {num_correct} correct attempts and {num_attempts} total "
            "attempts.")
    
    if k <= 0:
        raise ValueError(f"K must be a positive integer. Got {k}.")
    
    if num_correct == 0:
        return 0.0
    
    if num_attempts <= k:
        return 1.0

    p_choosing_all_incorrect: float = (
        math.comb(num_attempts - num_correct, k) / math.comb(num_attempts, k))

    return 1.0 - p_choosing_all_incorrect
