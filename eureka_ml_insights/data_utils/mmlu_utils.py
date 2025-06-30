from dataclasses import dataclass

import pandas as pd

from .transform import DFTransformBase

# The list of 57 tasks is taken from `https://github.com/hendrycks/test/blob/master/categories.py`

MMLUCategories = {
    "STEM": [
        "astronomy",
        "college_physics",
        "conceptual_physics",
        "high_school_physics",
        "college_chemistry",
        "high_school_chemistry",
        "college_biology",
        "high_school_biology",
        "college_computer_science",
        "computer_security",
        "high_school_computer_science",
        "machine_learning",
        "abstract_algebra",
        "college_mathematics",
        "elementary_mathematics",
        "high_school_mathematics",
        "high_school_statistics",
        "electrical_engineering",
    ],
    "Humanities": [
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "prehistory",
        "formal_logic",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "world_religions",
        "international_law",
        "jurisprudence",
        "professional_law",
    ],
    "Social Sciences": [
        "high_school_government_and_politics",
        "public_relations",
        "security_studies",
        "us_foreign_policy",
        "human_sexuality",
        "sociology",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "econometrics",
        "high_school_geography",
        "high_school_psychology",
        "professional_psychology",
    ],
    "Other (Business, Health, Misc.)": [
        "global_facts",
        "miscellaneous",
        "professional_accounting",
        "business_ethics",
        "management",
        "marketing",
        "anatomy",
        "clinical_knowledge",
        "college_medicine",
        "human_aging",
        "medical_genetics",
        "nutrition",
        "professional_medicine",
        "virology",
    ],
}

MMLUTaskToCategories = {task: cat for cat, tasks in MMLUCategories.items() for task in tasks}

MMLUAll = [task for cat in MMLUCategories.values() for task in cat]


@dataclass
class CreateMMLUPrompts(DFTransformBase):
    """Transform to create prompts for MMLU dataset."""
    def __init__(self):
        self.multi_option_example_format = "{}\n{}\nAnswer with the option's letter from the given options directly."

    def _create_prompt(self, sample):
        question = sample["question"]
        options = sample["choices"]
        example = ""
        start_chr = "A"
        index2ans = {}

        for option in options:
            example += f"({start_chr}) {option}\n"
            index2ans[start_chr] = option
            start_chr = chr(ord(start_chr) + 1)

        prompt = self.multi_option_example_format.format(question, example)

        return prompt

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df["prompt"] = df.apply(self._create_prompt, axis=1)

        return df
