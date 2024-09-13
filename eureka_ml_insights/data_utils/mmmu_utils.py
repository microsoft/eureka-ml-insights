from dataclasses import dataclass

import pandas as pd

from eureka_ml_insights.data_utils import DFTransformBase

MMMUCategories = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": [
        "Biology",
        "Chemistry",
        "Geography",
        "Math",
        "Physics",
    ],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": ["History", "Literature", "Sociology", "Psychology"],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}

MMMUTaskToCategories = {task: cat[0] for cat in MMMUCategories.items() for task in cat[1]}

MMMUAll = [task for cat in MMMUCategories.values() for task in cat]


@dataclass
class CreateMMMUPrompts(DFTransformBase):
    """
    Create prompts in the MMMU format for mutiple-choice and open-ended questions
    The original code is located in https://github.com/MMMU-Benchmark/MMMU/tree/main/eval/utils and has
    small modifications to work in this framework.
    """

    def __init__(self):
        self.multi_choice_example_format = "{}\n{}\nAnswer with the option's letter from the given choices directly."
        self.short_ans_example_format = "{}\nAnswer the question using a single word or phrase."

    def _create_prompt(self, sample):
        question = sample["question"]
        options = sample["options"]
        example = ""
        if sample["question_type"] == "multiple-choice":
            start_chr = "A"
            prediction_range = []
            index2ans = {}
            for option in options:
                prediction_range.append(start_chr)
                example += f"({start_chr}) {option}\n"
                index2ans[start_chr] = option
                start_chr = chr(ord(start_chr) + 1)

            prompt = self.multi_choice_example_format.format(question, example)
        else:
            prompt = self.short_ans_example_format.format(question)

        return prompt

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df["prompt"] = df.apply(self._create_prompt, axis=1)

        return df
