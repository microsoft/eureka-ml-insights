"""Defines a transform that decodes test cases.

Reproduces the behavior of the CodeGenerationProblem.__post_init__ method
within lcb_runner/benchmarks/code_generation.py in the LiveCodeBench
repository.
"""

import pandas as pd
import dataclasses

from tqdm.auto import tqdm
from typing import override

from eureka_ml_insights.data_utils import transform
from eureka_ml_insights.data_utils.live_code_bench import encoding

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

    @override
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
                encoding.decode_test_cases))
        return df
