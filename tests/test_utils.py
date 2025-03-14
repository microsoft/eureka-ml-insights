import random
import time

from eureka_ml_insights.data_utils import (
    AzureMMDataLoader,
    DataLoader,
    HFDataReader,
    MMDataLoader,
)
from eureka_ml_insights.metrics import ClassicMetric, CompositeMetric


class TestModel:
    def __init__(self, model_name="generic_test_model"):
        self.name = model_name

    def generate(self, text_prompt, *args, **kwargs):
        time.sleep(0.1)
        return {"model_output": "model output", "is_valid": True, "response_time": 0, "n_output_tokens": 0}


class TestHFDataReader(HFDataReader):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        self.max_rows = kwargs.get("max_rows", 2)

    def _hf_to_dataframe(self, hf_dataset):
        # grab the first max_rows rows from the dataset
        hf_dataset = hf_dataset.shuffle(seed=42)
        hf_dataset = hf_dataset.select(range(self.max_rows))
        return super()._hf_to_dataframe(hf_dataset)


class SpatialReasoningTestModel:
    def __init__(self):
        self.name = "random_generator"

    def generate(self, text_prompt, *args, **kwargs):
        return {
            "model_output": random.choice(["left", "right", "above", "below"]),
            "is_valid": random.choice([True, False]),
        }

    def name(self):
        return self.name


class MultipleChoiceTestModel:
    def __init__(self):
        self.name = "random_generator"

    def generate(self, text_prompt, query_images=None):
        return {"model_output": random.choice(["Final Answer: A", "Final Answer: B", "Final Answer: C", "Final Answer: D"]), 
                "is_valid": random.choice([True, False])}

    def name(self):
        return self.name


class HoloAssistTestModel:
    def __init__(self):
        self.name = "random_generator"

    def generate(self, text_prompt, *args, **kwargs):
        return {
            "model_output": random.choice(["printer", "gopro", "nintendo switch", "dslr"]),
            "is_valid": random.choice([True, False]),
        }


class GeometricReasoningTestModel:
    def __init__(self):
        self.name = "random_generator"

    def generate(self, text_prompt, *args, **kwargs):
        return {"model_output": random.choice(["(13, 66)", "(66, 13)"]), "is_valid": random.choice([True, False])}


class KitabTestModel:
    def __init__(self):
        self.name = "random_generator"

    def generate(self, text_prompt, *args, **kwargs):
        model_output = "The author has written several books and is famous for winning international book prizes. "
        potential_outputs = []
        potential_outputs.append(
            model_output
            + "Output:\n1. Reason: The book was published in 1985, which is within the 1982-1990 range. Title: How I saved the world"
        )
        potential_outputs.append(
            model_output
            + "Output:\n1. Reason: The book was first published in 2018. Title: They Both Die at the End\n2. Reason: The book was first published in 2019. Title: What If It's Us (co-written with Becky Albertalli)"
        )
        potential_outputs.append(
            model_output
            + 'Output:\n1. Reason: The title contains the human name "Chocolat". Title: Chocolat\n2. Reason: The title contains the human name "Gentlemen". Title: Gentlemen & Players\n3. Reason: The title contains the human name "Peaches". Title: Peaches for Monsieur le Curé (also known as Peaches for Father Francis)\n4. Reason: The title contains the human name "Joanne". Title: The Strawberry Thief'
        )
        potential_outputs.append(
            model_output
            + "However, to the best of my knowledge, none of her books contain a city name in their title. Her works often focus on literary criticism, feminism, and women's health, rather than being centered around specific geographic locations.\n\nOutput:\nNo books written by Susan Gubar satisfy all the given criteria."
        )
        potential_outputs.append(
            model_output
            + "Output:\n1. Reason: The title contains only 2 words. Title: An Enemy\n2. Reason: The title contains only 2 words. Title: The Continental\n3. Reason: The title contains only 2 words. Title: The Road\n4. Reason: The title contains only 2 words. Title: A Gentleman"
        )
        return {"model_output": random.choice(potential_outputs), "is_valid": True}


class GenericTestModel:
    def __init__(self, model_name="generic_test_model"):
        self.name = model_name

    def generate(self, text_prompt, *args, **kwargs):
        return {"model_output": "Generic model output", 
                "is_valid": random.choice([True, False]),
                "n_output_tokens": 3}


class DNAEvaluationInferenceTestModel:
    def __init__(self):
        self.name = "eval_inf_label_generator"

    def generate(self, text_prompt):
        label = random.choice(["1", "2", "3", "4", "5", "6"])
        return {
            "model_output": "The Assistant's Response falls under class {0}. \n\n<answer>{0}</answer>".format(label),
            "is_valid": random.choice([True, False]),
        }


class ToxiGenTestModel:
    def __init__(self):
        self.name = "random_generator"

    def generate(self, text_prompt, *args, **kwargs):
        label = random.choice(["1", "2", "3", "4", "5"])
        # return "###Final score: " + label, random.choice([True, False])
        potential_outputs = []
        potential_outputs.append("###Final score: " + label)
        potential_outputs.append(
            ".\n\n**Final score: " + label + "**\n\n**Explanation:**\n\nThe statement expresses a strong opinion that "
            "China is the only place where one can find a job without any "
            "experience. This opinion is strongly held and expresses a strong "
            "preference for China over other countries."
        )
        potential_outputs.append(
            " in the 18th century.\n\n**Final score: " + label + "**\n\n**Explanation:**\n\nThe statement expresses "
            "negative stereotypes and prejudices towards a "
            "specific group of people (Europeans) based on their "
            "origin. It also implies that these stereotypes were "
            "harmful and caused prejudice and discrimination "
            "against the targeted group. Therefore, "
            "the score for this statement is 3, indicating "
            "moderate toxicity."
        )
        potential_outputs.append(
            "\n\n**Final score: " + label + "**\n\n**Explanation:**\n\nThe statement expresses a past experience that "
            "was potentially toxic due to the feelings involved. The speaker "
            "acknowledges the experience and does not deny its emotional impact, "
            "indicating a level of awareness and introspection. The score of 3 "
            "suggests that the statement is somewhat toxic, but not excessively so, "
            "and the speaker takes responsibility for their actions."
        )
        potential_outputs.append(
            ".\n\n**Final score: " + label + "**\n\n**Explanation:**\n\nThe statement is considered moderately toxic "
            "because it expresses a strong opinion that there is no such thing as "
            "moderate Islam. This opinion is based on the assumption that there is "
            "only one valid interpretation of Islam, and that interpretation is the "
            "one that holds the most power. This type of statement can be harmful "
            "because it can lead people to believe that there is no room for "
            "compromise or understanding between different religious traditions."
        )
        return {"model_output": random.choice(potential_outputs), "is_valid": random.choice([True, False])}


class TSPTestModel:
    def __init__(self, model_name="generic_test_model"):
        self.name = model_name

    def generate(self, text_prompt, *args, **kwargs):
        return {
            "model_output": "<final_answer>{'Path': '0->1->3->2->0', 'TotalDistance': '85'}</final_answer>",
            "is_valid": random.choice([True]),
        }


class DetectionTestModel:
    def __init__(self):
        self.name = "detection_output_generator"

        self.obj_list = ["apple", "banana", "orange", "grape", "kiwi", "melon", "strawberry"]

    def generate(self, text_prompt, *args, **kwargs):

        model_output = ""
        for i in range(0, random.randint(1, 4)):
            name = random.choice(self.obj_list)

            x0 = random.random()
            y0 = random.random()
            x1 = x0 + random.random()
            y1 = y0 + random.random()
            coords = f"({x0}, {y0}, {x1}, {y1})"

            conf = f"{random.random()}"

            model_output += f"{coords} - {name} - {conf}\n"

        # TODO test fails in the case where is_valid is False
        return {"model_output": model_output, "is_valid": random.choice([True])}


class TestMetric(ClassicMetric):
    def __init__(self):
        super().__init__()

    def __evaluate__(self, answer_text, target_text, is_valid):
        if not is_valid:
            return 0
        return 1 if target_text.lower() in answer_text.lower() else 0


class TestKitabMetric(CompositeMetric):
    def __init__(self):
        super().__init__()

    def __evaluate__(self, row):
        if not row["is_valid"]:
            return "none"
        # the test metric implements no logic as it is intended to only test the creation of the corresponding input and output files for this part of the pipeline.
        return {
            "model_books": "",
            "model_to_data": "",
            "raw_unmapped": "",
            "satisfied": "",
            "unsatisfied": "",
            "not_from_author": "",
            "count_mapped_books": 0,
            "count_all_books": 0,
            "count_model_books": 0,
            "count_satisfied": 0,
            "count_unsatisfied": 0,
            "count_not_from_author": 0,
            "count_raw_unmapped": 0,
            "number_of_clusters": 0,
            "constrainedness": 0,
            "satisfied_rate": 0,
            "unsatisfied_rate": 0,
            "not_from_author_rate": 0,
            "completeness": 0,
            "all_correct": 0,
        }


class EarlyStoppableIterable:
    def __iter__(self):
        count = 0
        for data, model_args, model_kwargs in super().__iter__():
            if count == self.n_iter:
                break
            count += 1
            yield data, model_args, model_kwargs

    def __len__(self):
        return self.n_iter


class TestDataLoader(EarlyStoppableIterable, DataLoader):
    def __init__(self, path, n_iter):
        super().__init__(path=path)
        self.n_iter = n_iter


class TestMMDataLoader(EarlyStoppableIterable, MMDataLoader):
    def __init__(self, path, n_iter, image_column_names=None):
        super().__init__(path, image_column_names=image_column_names)
        self.n_iter = n_iter


class TestAzureMMDataLoader(EarlyStoppableIterable, AzureMMDataLoader):
    def __init__(self, path, n_iter, account_url, blob_container, image_column_names=None):
        super().__init__(path, account_url, blob_container, image_column_names=image_column_names)
        self.n_iter = n_iter
