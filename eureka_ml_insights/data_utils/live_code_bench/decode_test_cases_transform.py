"""Defines a transform that decodes test cases."""

import dataclasses

from eureka_ml_insights.data_utils import transform

from eureka_ml_insights.data_utils.live_code_bench import encoding

@dataclasses.dataclass
class DecodeTestCasesTransform(transform.DFTransformBase):
    """Decodes the test cases from the model output."""

    encoded_test_cases_column_name: str
    decoded_test_cases_column_name: str

    def transform(self, df):
        df[self.decoded_test_cases_column_name] = (
            df[self.encoded_test_cases_column_name].apply(
                encoding.decode_test_cases))
        return df
