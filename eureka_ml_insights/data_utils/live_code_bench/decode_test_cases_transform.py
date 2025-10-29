"""Defines a transform that decodes test cases.

Reproduces the behavior of the CodeGenerationProblem.__post_init__ method
within https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/benchmarks/code_generation.py#L65
"""

import pandas as pd
import dataclasses
import base64
import json
import pickle
import zlib

from typing import Any

from tqdm.auto import tqdm

from eureka_ml_insights.data_utils import transform


@dataclasses.dataclass
class DecodeTestCasesTransform(transform.DFTransformBase):
    """Decodes the test cases from the model output.

    Attributes:
        encoded_test_cases_column_name: The name of the column containing the
            encoded test cases.
        decoded_test_cases_column_name: The name of the column to store the
            decoded test cases.
    """

    encoded_test_cases_column_name: str
    decoded_test_cases_column_name: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decodes the test cases in the DataFrame.

        Args:
            df: The input DataFrame.

        Returns:
            The DataFrame with the decoded test cases.
        """
        tqdm.pandas(desc=f"Decoding {self.encoded_test_cases_column_name}")

        df[self.decoded_test_cases_column_name] = (
            df[self.encoded_test_cases_column_name].progress_apply(
                decode_test_cases))
        return df


def encode_test_cases(test_cases: Any, compress: bool = False) -> str:
    """Encodes test cases either as JSON or compressed base64-encoded pickled JSON.

    If ``compress`` is True, the test cases are serialized as JSON, pickled,
    compressed with zlib, and base64-encoded. Otherwise, they are serialized
    directly as JSON.

    Args:
        test_cases: The test cases to encode.
        compress: Whether to use compression and base64 encoding.

    Returns:
        The encoded test cases as a string.

    Raises:
        ValueError: If encoding fails.
    """
    try:
        if compress:
            # Convert to JSON first to ensure the data is serializable
            json_bytes = json.dumps(test_cases).encode("utf-8")
            pickled = pickle.dumps(json_bytes)
            compressed = zlib.compress(pickled)
            base64_encoded = base64.b64encode(compressed)
            return base64_encoded.decode("utf-8")
        else:
            return json.dumps(test_cases)
    except Exception as e:
        raise ValueError(
            f"Failed to encode test cases: {e.__class__.__name__}: {e}") from e


def decode_test_cases(encoded_test_cases: str | None) -> Any:
    """Decodes test cases either from JSON or compressed base64-encoded pickled JSON.

    Tries to decode as JSON first. If that fails, attempts to decode as
    compressed base64-encoded pickled JSON.

    Args:
        encoded_test_cases: The encoded test cases as a string.

    Returns:
        The decoded test cases.

    Raises:
        ValueError: If decoding fails for both JSON and compressed formats.
    """
    if encoded_test_cases is None:
        return None

    try:
        return json.loads(encoded_test_cases)
    except json.JSONDecodeError as e_json:
        json_error = e_json

    try:
        base64_decoded = base64.b64decode(encoded_test_cases.encode("utf-8"))
        decompressed = zlib.decompress(base64_decoded)
        return json.loads(pickle.loads(decompressed))
    except Exception as e_compressed:
        compressed_error = e_compressed

    raise ValueError(
        "Failed to decode test cases. Tried JSON and compressed base64 formats.\n"
        f"- JSON decoding error: {json_error}\n"
        "- Compressed decoding error: "
        f"{compressed_error.__class__.__name__}: {compressed_error}"
    )
