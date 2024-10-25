from dataclasses import dataclass

import pandas as pd

import random

from eureka_ml_insights.data_utils import DFTransformBase

@dataclass
class CreateGPQAPrompt(DFTransformBase):
    """
    Creates the prompt for GPQA.
    """
    def __init__(self):
        self.direct_answer = "Answer the following question by saying 'My answer is <letter of your answer choice>.' Don't provide any explanations or other information."

    def _rand_map_answers_to_mc(self):
        # List of the column names to shuffle
        columns_to_shuffle = ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]

        # Shuffle the column names
        shuffled_columns = random.sample(columns_to_shuffle, len(columns_to_shuffle))

        # Map shuffled columns to A, B, C, D and store the mapping
        mapping = {
            "A": shuffled_columns[0],
            "B": shuffled_columns[1],
            "C": shuffled_columns[2],
            "D": shuffled_columns[3]
        }

        return mapping
    
    def _create_prompt(self, sample):
        question = sample["Question"]
        answer_mapping = self._rand_map_answers_to_mc()
        prompt = self.direct_answer + "\n" + question
        gold_answer_label = ""

        for key in answer_mapping:
            possible_ans = sample[answer_mapping[key]]
            if answer_mapping[key] == "Correct Answer":
                gold_answer_label = key
            prompt += f"{key}) {possible_ans}\n"

        return [prompt, gold_answer_label]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        #print(df.apply(lambda row: self._create_prompt(row), axis=1))
        df[["prompt", "ground_truth"]] = df.apply(lambda row: pd.Series(self._create_prompt(row)), axis=1)
        
        return df