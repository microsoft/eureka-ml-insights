import re
from dataclasses import dataclass

import pandas as pd
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
)

from .transform import DFTransformBase


@dataclass
class BFCLMultiturnExecuteCall(DFTransformBase):
    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.model_answer_column] = df.apply(self.execuate_model_output, axis=1)
        return df

    @staticmethod
    def execuate_model_output(df_row):
        """
        Execute the model output to get the function output.

        Parameters:
            df_row (a row in dataframe): Input is a dataframe row containing model_output, initial_config, and, involved_classes.
        Returns:
            " ".join(execution_results) (str): A string denoting the execution of the function calls.
        """
        test_entry = df_row
        response_text = test_entry["model_output"]
        initial_config: dict = eval(test_entry["initial_config"])
        involved_classes: list = eval(test_entry["involved_classes"])
        test_entry_id: str = test_entry["id"]
        test_category: str = test_entry_id.rsplit("_", 1)[0]

        func_calls = re.findall(r"\w+\([^)]*\)", response_text)
        if len(func_calls) == 0:
            return "No call executed"

        execution_results, involved_instances = execute_multi_turn_func_call(
            func_call_list=func_calls,
            initial_config=initial_config,
            involved_classes=involved_classes,
            model_name="",
            test_entry_id=test_entry_id,
            long_context=("long_context" in test_category or "composite" in test_category),
            is_evaL_run=False,
        )
        return " ".join(execution_results)
