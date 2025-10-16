"""Defines some transformation utilities for the LiveCodeBench benchmark."""

import base64
import dataclasses
import json
import pickle
import re
import zlib

from eureka_ml_insights.data_utils import transform


def _extract_code(response: str | None) -> str | None:
    if response is None:
        return None

    # Try to find a code snippet in markdown format
    match = re.search(r"```(?:python)?\n(.*?)\n```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return ""


def _decode_compressed_base64_json(encoded_test_cases: str) -> str:
    base64_decoded = base64.b64decode(encoded_test_cases.encode("utf-8"))
    decompressed = zlib.decompress(base64_decoded)
    return json.loads(pickle.loads(decompressed))


def _decode_test_cases(encoded_test_cases: str | None) -> str | None:
    if encoded_test_cases is None:
        return None

    try:
        return json.loads(encoded_test_cases)
    except json.JSONDecodeError:
        pass

    return _decode_compressed_base64_json(encoded_test_cases)


@dataclasses.dataclass
class CodeExtractionTransform(transform.DFTransformBase):
    """Extracts the code snippet from the model output."""

    model_output_column: str
    code_column: str

    def transform(self, df):
        df[self.code_column] = (
            df[self.model_output_column].apply(_extract_code))
        return df


@dataclasses.dataclass
class DecodeTestCasesTransform(transform.DFTransformBase):
    """Decodes the test cases from the model output."""

    encoded_test_cases_column_name: str
    decoded_test_cases_column_name: str

    def transform(self, df):
        df[self.decoded_test_cases_column_name] = (
            df[self.encoded_test_cases_column_name].apply(_decode_test_cases))
        return df
