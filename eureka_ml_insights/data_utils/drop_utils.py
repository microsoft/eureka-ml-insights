from dataclasses import dataclass

import pandas as pd

import random

from eureka_ml_insights.data_utils import DFTransformBase

@dataclass
class CreateDropPrompt(DFTransformBase):
    """
    Creates the prompt for Drop.
    """
    def __init__(self):
        self.instruction = "Use the following passage to answer the question. End your response by saying 'My answer is...'."
    
    def _create_prompt(self, sample):
        passage = sample["passage"]
        question = sample["question"]
        answer_spans = sample["answers_spans"]
        
        prompt = f"{self.instruction}\nPassage:{passage}\nQuestion:{question}"
        answers = answer_spans["spans"]
        return prompt, answers

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        #print(df.apply(lambda row: self._create_prompt(row), axis=1))
        df[["prompt", "ground_truth"]] = df.apply(lambda row: pd.Series(self._create_prompt(row)), axis=1)
        
        return df