import re
import string
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .transform import DFTransformBase


@dataclass
class FlenQAOutputProcessor(DFTransformBase):
    """
    This class is for processing the output of models for the FLenQA dataset.
    """

    chain_of_thought: bool = False

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ground_truth"] = df["ground_truth"].apply(lambda x: x.lower())
        processed_columns = df.apply(
            lambda row: response_analysis(
                row["model_output"],
                row["dataset"],
                row["facts"],
                row["statement"],
                row["is_valid"],
                self.chain_of_thought,
            ),
            axis=1,
        )
        df = pd.concat([df, processed_columns], axis=1)
        return df


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra white spaces
    Code taken from: https://github.com/alonj/Same-Task-More-Tokens

    Args:
    s: string to normalize

    Returns:
    normalized string
    """
    s = str(s).lower()
    s = s.replace("".join(list(set(string.punctuation))), "")
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def response_category(ans):
    """
    Categorize the answer as true, false or other/refused
    Code taken from: https://github.com/alonj/Same-Task-More-Tokens
    Args:
    ans: string to categorize

    Returns:
    string category
    """
    if isinstance(ans, (bool, np.bool_)):
        return normalize_answer(str(ans))
    if isinstance(ans, str):
        ans = normalize_answer(ans)
        ans = ans.replace("not true", "false")
        last_true_pos = ans.rfind("true")
        last_false_pos = ans.rfind("false")
        if last_true_pos > last_false_pos:
            return "true"
        elif last_false_pos > last_true_pos:
            return "false"
    return "none"


def response_analysis(response, dataset, facts, statement, is_valid, chain_of_thought=False):
    """
    Analyze the model response, calculate chain of thought coverage, response category and early response.
    Code taken and adapted from: https://github.com/alonj/Same-Task-More-Tokens

    Args:
        response: model output to analyze
        dataset: dataset name which is a column in the original flenQA dataset
        facts: facts column from the original flenQA dataset
        statement: statement column from the original flenQA dataset
        is_valid: boolean column output from inference component indicating if the response is valid
        chain_of_thought: boolean indicating if a chain of thought prompt was used.

    Returns:
        Pandas series containing analysis results.
    """
    column_names = ["cot_coverage", "categorical_response", "early_response", "is_valid"]
    if not is_valid:
        return pd.Series([0, "none", False, False], index=column_names)
    normalized_response_text = normalize_answer(response)
    categorical_response = response_category(normalized_response_text)
    if chain_of_thought:
        if dataset != "Simplified RuleTaker":  # Ruletaker has statements instead of facts
            cot_coverage = sum([normalize_answer(fact) in normalized_response_text for fact in facts])
        else:
            cot_coverage = sum([normalize_answer(fact) in normalized_response_text for fact in statement])
        early_response = (
            categorical_response is not None and categorical_response in normalized_response_text[:10].lower()
        )
    else:
        cot_coverage = 0
        early_response = False
    return pd.Series([cot_coverage, categorical_response, early_response, True], index=column_names)
