import re
import string
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .transform import DFTransformBase


@dataclass
class FlenQAOutputProcessor(DFTransformBase):
    """
    Handles processing the output of models for the FLenQA dataset.

    Attributes:
        chain_of_thought (bool): If True, the chain of thought is used in the analysis.
    """

    chain_of_thought: bool = False

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms a given DataFrame for the FLenQA dataset.

        Args:
            df (pd.DataFrame): The input DataFrame containing model output.

        Returns:
            pd.DataFrame: The transformed DataFrame with additional columns.
        """
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
    Lowers the text, removes punctuation, articles, and extra white spaces.

    This code is adapted from:
    https://github.com/alonj/Same-Task-More-Tokens

    Args:
        s (str): The string to normalize.

    Returns:
        str: The normalized string.
    """
    s = str(s).lower()
    s = s.replace("".join(list(set(string.punctuation))), "")
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def response_category(ans):
    """
    Categorizes the answer as "true," "false," or "none."

    This code is adapted from:
    https://github.com/alonj/Same-Task-More-Tokens

    Args:
        ans (str): The string to categorize.

    Returns:
        str: The category of the answer.
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
    Analyzes the model response by calculating chain of thought coverage, response category, and early response.

    This code is taken and adapted from:
    https://github.com/alonj/Same-Task-More-Tokens

    Args:
        response (str): The model output to analyze.
        dataset (str): The dataset name which is a column in the original flenQA dataset.
        facts (list): The facts column from the original flenQA dataset.
        statement (list): The statement column from the original flenQA dataset.
        is_valid (bool): Indicates if the response is valid.
        chain_of_thought (bool): Indicates if a chain of thought prompt was used. Defaults to False.

    Returns:
        pd.Series: A pandas Series containing analysis results.
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