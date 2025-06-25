"""Module that provides MMMUMetric class for evaluating multiple-choice and open-ended questions from the MMMU dataset."""

import random
import re

import numpy as np

from .metrics_base import MixedQuestionTypeMetric


class MMMUMetric(MixedQuestionTypeMetric):
    """Implements the exact metric from the MMMU dataset for multiple-choice and open-ended answers.

    The original code is located at:
    https://github.com/MMMU-Benchmark/MMMU/tree/main/eval/utils
    and includes small modifications to work in this framework.
    """

    def __init__(self):
        """Initializes MMMUMetric by seeding the random number generator with 42."""
        super().__init__()

        random.seed(42)

    # ----------- Process Multi-choice -------------
    def parse_multi_choice_response(self, response, all_choices, index2ans):
        """Parses the prediction from the generated response and returns the predicted index.

        This method validates the response content to identify the best matching multiple-choice index
        (e.g., 'A', 'B', 'C', 'D'). If no matching index is found, one is randomly chosen.

        Args:
            response (str): The model-generated response.
            all_choices (List[str]): A list of possible choice indices (e.g., ['A', 'B', 'C', 'D']).
            index2ans (Dict[str, str]): A mapping from choice indices to full answer strings.

        Returns:
            str: The predicted choice index (e.g., 'A', 'B', 'C', or 'D').
        """
        for char in [",", ".", "!", "?", ";", ":", "'"]:
            response = response.strip(char)
        response = " " + response + " "  # add space to avoid partial match

        index_ans = True
        ans_with_brack = False
        candidates = []
        for choice in all_choices:  # e.g., (A) (B) (C) (D)
            if f"({choice})" in response:
                candidates.append(choice)
                ans_with_brack = True

        if len(candidates) == 0:
            for choice in all_choices:  # e.g., A B C D
                if f" {choice} " in response:
                    candidates.append(choice)

        # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
        if len(candidates) == 0 and len(response.split()) > 5:
            for index, ans in index2ans.items():
                if ans.lower() in response.lower():
                    candidates.append(index)
                    index_ans = False  # it's content ans.

        if len(candidates) == 0:  # still not get answer, randomly choose one.
            pred_index = random.choice(all_choices)
        elif len(candidates) > 1:
            start_indexes = []
            if index_ans:
                if ans_with_brack:
                    for can in candidates:
                        index = response.rfind(f"({can})")
                        start_indexes.append(index)  # -1 will be ignored anyway
                else:
                    for can in candidates:
                        index = response.rfind(f" {can} ")
                        start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.lower().rfind(index2ans[can].lower())
                    start_indexes.append(index)
            # get the last one
            pred_index = candidates[np.argmax(start_indexes)]
        else:  # if only one candidate, use it.
            pred_index = candidates[0]

        return pred_index

    # ----------- Process Open -------------
    def check_is_number(self, string):
        """Checks whether the given string can be interpreted as a float.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the string can be converted to a float, False otherwise.
        """
        try:
            float(string.replace(",", ""))
            return True
        except ValueError:
            # check if there's comma inside
            return False

    def normalize_str(self, string):
        """Normalizes a string to lowercase or converts it to a float if it looks like a number.

        If the string is a number, it is converted to float and rounded to two decimals.
        If it is a single character, variants with leading and trailing spaces are returned
        to avoid trivial matches.

        Args:
            string (str): The string to normalize.

        Returns:
            list: A list containing either the normalized string(s) or float value(s).
        """
        string = string.strip()

        is_number = self.check_is_number(string)

        if is_number:
            string = string.replace(",", "")
            string = float(string)
            # leave 2 decimal
            string = round(string, 2)
            return [string]
        else:  # it's likely to be a string
            string = string.lower()
            if len(string) == 1:
                return [" " + string, string + " "]  # avoid trivial matches
            return [string]

    def extract_numbers(self, string):
        """Extracts various forms of numbers from a string using regular expressions.

        This includes numbers with commas, scientific notation, and simple numeric formats.

        Args:
            string (str): The string from which to extract numbers.

        Returns:
            list: A list of all extracted numeric substrings.
        """
        pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
        pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
        pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"

        numbers_with_commas = re.findall(pattern_commas, string)
        numbers_scientific = re.findall(pattern_scientific, string)
        numbers_simple = re.findall(pattern_simple, string)

        all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
        return all_numbers

    def parse_open_response(self, response):
        """Parses the prediction from a generated response and returns a list of predicted strings or numbers.

        This method extracts key sub-responses, numerical values, and normalizes them.
        Duplicate values are removed.

        Args:
            response (str): The model-generated response.

        Returns:
            list: A list of predicted strings or numbers.
        """

        def get_key_subresponses(response):
            key_responses = []
            response = response.strip().strip(".").lower()
            sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)
            indicators_of_keys = ["could be ", "so ", "is ", "thus ", "therefore ", "final ", "answer ", "result "]
            key_responses = []
            for index, resp in enumerate(sub_responses):
                if index == len(sub_responses) - 1:
                    indicators_of_keys.extend(["="])
                shortest_key_response = None
                for indicator in indicators_of_keys:
                    if indicator in resp:
                        if not shortest_key_response:
                            shortest_key_response = resp.split(indicator)[-1].strip()
                        else:
                            if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                                shortest_key_response = resp.split(indicator)[-1].strip()

                if shortest_key_response:
                    if shortest_key_response.strip() not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                        key_responses.append(shortest_key_response)
            if len(key_responses) == 0:
                return [response]
            return key_responses

        key_responses = get_key_subresponses(response)

        pred_list = key_responses.copy()
        for resp in key_responses:
            pred_list.extend(self.extract_numbers(resp))

        tmp_pred_list = []
        for i in range(len(pred_list)):
            tmp_pred_list.extend(self.normalize_str(pred_list[i]))
        pred_list = tmp_pred_list

        pred_list = list(set(pred_list))

        return pred_list

    # ----------- Evaluation -------------
    def eval_multi_choice(self, gold_i, pred_i):
        """Evaluates the correctness of a multiple-choice prediction.

        Args:
            gold_i (str or list of str): The correct answer(s). May be a single string or a list of valid answers.
            pred_i (str): The predicted choice index.

        Returns:
            bool: True if the prediction matches any of the valid answers, False otherwise.
        """
        correct = False
        if isinstance(gold_i, list):
            for answer in gold_i:
                if answer == pred_i:
                    correct = True
                    break
        else:
            if gold_i == pred_i:
                correct = True
        return correct

    def eval_open(self, gold_i, pred_i):
        """Evaluates the correctness of an open-ended response.

        The gold answers and predicted answers are normalized for comparison.

        Args:
            gold_i (str or list of str): The correct answer(s).
            pred_i (list of (str or float)): The predicted answers, previously normalized.

        Returns:
            bool: True if any of the predictions matches the correct answer(s), False otherwise.
        """
        correct = False
        if isinstance(gold_i, list):
            norm_answers = []
            for answer in gold_i:
                norm_answers.extend(self.normalize_str(answer))
        else:
            norm_answers = self.normalize_str(gold_i)
        for pred in pred_i:
            if isinstance(pred, str):
                for norm_ans in norm_answers:
                    if isinstance(norm_ans, str) and norm_ans in pred:
                        if not correct:
                            correct = True
                        break
            else:
                if pred in norm_answers:
                    if not correct:
                        correct = True
                    break
        return correct

    def get_multi_choice_info(self, options):
        """Generates a mapping and list of choice indices for multiple-choice questions.

        Args:
            options (list of str): A list of multiple-choice options.

        Returns:
            tuple: A tuple (index2ans, all_choices) where:
                index2ans (dict): Maps a choice index (e.g., 'A') to the corresponding option text.
                all_choices (list of str): A list of choice indices (e.g., ['A', 'B', 'C', ...]).
        """
        start_chr = "A"
        all_choices = []
        index2ans = {}
        for i, option in enumerate(options):
            index2ans[chr(ord(start_chr) + i)] = option
            all_choices.append(chr(ord(start_chr) + i))

        return index2ans, all_choices

    def __evaluate__(self, answer_text, target_text, question_type, target_options, is_valid):
        """Evaluates a single question instance to determine correctness.

        For a multiple-choice question, it parses the response to find a predicted index
        and compares it against the gold answer. For an open-ended question, it uses the
        parsed open response and compares it to the gold answer.

        Args:
            answer_text (str): The model-generated answer text.
            target_text (str or list of str): The correct answer(s).
            question_type (str): The type of question, either 'multiple-choice' or 'open'.
            target_options (list of str): The multiple-choice options if applicable.
            is_valid (bool): Whether this question should be evaluated.

        Returns:
            str: "none" if the question is invalid, "correct" if the prediction is correct, or "incorrect" otherwise.
        """
        if not is_valid:
            return "none"

        if question_type == "multiple-choice":
            index2ans, all_choices = self.get_multi_choice_info(target_options)
            parsed_pred = self.parse_multi_choice_response(answer_text, all_choices, index2ans)
            correct = self.eval_multi_choice(target_text, parsed_pred)
        else:  # open question
            parsed_pred = self.parse_open_response(answer_text)
            correct = self.eval_open(target_text, parsed_pred)

        return "correct" if correct else "incorrect"