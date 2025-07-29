import re, json, ast
from dataclasses import dataclass

import pandas as pd

from .transform import DFTransformBase

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    STATELESS_CLASSES,
    execute_multi_turn_func_call,
    is_empty_execute_response,
)

@dataclass
class BFCLMultiturnExecuteCall(DFTransformBase):
    model_output_column: str
    model_answer_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.model_answer_column] = df.apply(self.execuate_model_output,axis=1)
#        df[self.model_answer_column] = df[self.model_output_column].apply(self.execuate_model_output)
        return df

    @staticmethod
    def execuate_model_output(response):
        """
        Execute the model output to get the function output.
             
        Parameters:
            response (str): Input string containing answer X in the form of "Final Answer: X".
        Returns:
            numerical_value (float or str): A numeric value or JSON string representing the model's answer.
        """
        print(response)
        test_entry = response
        response_text = test_entry["model_output"]
        initial_config: dict = eval(test_entry["initial_config"])
        involved_classes: list = eval(test_entry["involved_classes"])
        test_entry_id: str = test_entry["id"]
        test_category: str = test_entry_id.rsplit("_", 1)[0]

        func_calls = re.findall(r'\w+\([^)]*\)', response_text)
        if(len(func_calls)==0):
            return "No call executed"
        print("start executing multi-turn func calls", func_calls)
        print(f"response_text {response_text}")
        print(f"initial_config {type(initial_config)} {initial_config}")
        print(f"involved_classes {type(initial_config)} {involved_classes}")
        print(f"test_entry_id {type(test_entry_id)}{test_entry_id}")
        print(f"test_category {type(test_category)} {test_category}")

        execution_results, involved_instances = execute_multi_turn_func_call(
        func_call_list = func_calls, 
        initial_config = initial_config,
        involved_classes = involved_classes,
        model_name = "",
        test_entry_id=test_entry_id,
        long_context = (
                        "long_context" in test_category or "composite" in test_category
                    ),
        is_evaL_run=False,
    )
        print("BFCL multi-turn result...",execution_results)
        return " ".join(execution_results)
        return extract_last_json_dict(response)
        # Remove <think>...</think> blocks
        response_cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        try:
            json.loads(response_cleaned)
            return response_cleaned  # valid JSON
        except json.JSONDecodeError:
            try:
                result = ast.literal_eval(response_cleaned)
                if isinstance(result, dict):
                    return json.dumps(result)
            except Exception:
                return None

        return None


import json
import re

def extract_last_json_dict(text):
    """
    Extract the last complete outermost JSON dictionary from the given text.
    Handles nested dictionaries and ignores other non-JSON content.
    
    Parameters:
        text (str): Input text that may contain JSON dictionary.
    Returns:
        str or None: The last valid JSON dict string, or None if not found.
    """
    # Remove <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    stack = []
    json_candidates = []
    start_idx = None

    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start_idx = i
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    candidate = text[start_idx:i+1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            json_candidates.append(candidate)
                    except Exception:
                        pass
                    start_idx = None

    return json_candidates[-1] if json_candidates else None
