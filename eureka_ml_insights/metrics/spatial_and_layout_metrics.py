"""This module provides a collection of metrics related to object detection, multiple-choice evaluation, 
and text analysis. It also offers utility functions for downloading NLTK resources and comparing words 
using WordNet path similarity.
"""

import ast
import logging
import re

import nltk
from pycocotools.coco import COCO

from ..data_utils.data import JsonReader
from .metrics_base import ClassicMetric, DetectionMetric, MultipleChoiceMetric


def download_nltk_resources():
    """Downloads the 'wordnet' corpus if it is not already installed.

    This function attempts to locate the 'wordnet' corpus. If it is not found,
    it downloads the corpus using nltk.download.
    """
    try:
        nltk.data.find("corpora/wordnet.zip")
    except LookupError:
        nltk.download("wordnet")


download_nltk_resources()
from nltk.corpus import wordnet


class SpatialAndLayoutReasoningMetric(MultipleChoiceMetric):
    """SpatialAndLayoutReasoningMetric requires a correct prediction to be only one valid multiple choice answer.

    This class inherits from MultipleChoiceMetric and checks whether only the correct option, among
    all the target options, is present exactly once in the answer text.
    """

    def __init__(self):
        """Initializes the SpatialAndLayoutReasoningMetric."""
        super().__init__()

    def __evaluate__(self, answer_text, target_text, target_options, is_valid):
        """Evaluates the answer against a list of valid target options.

        Args:
            answer_text (str): The text provided as the answer.
            target_text (str): The correct target option.
            target_options (list[str]): All possible valid choices.
            is_valid (bool): Indicates if the sample is valid for evaluation.

        Returns:
            str: Returns "correct" if exactly one option appears and it is the target_text,
            "incorrect" if exactly one option appears and it is not the target_text,
            otherwise "none".
        """
        if not is_valid:
            return "none"

        options_count = {}
        for option in target_options:
            # Ensure matching whole words by including a word boundary "\b" in the pattern
            pattern = "\\b{phrase}\\b".format(phrase=option)
            matches = re.findall(pattern, answer_text, flags=re.IGNORECASE)
            options_count[option] = len(matches)

        total_count = sum(options_count.values())

        # "correct" if only the right answer appears once,
        # "incorrect" if only a wrong answer appears,
        # "none" if there are multiple answers or no matches
        return (
            "correct"
            if (total_count == 1 and options_count[target_text] == 1)
            else "incorrect" if (total_count == 1) else "none"
        )


def wordnet_compare(word, compare_word, threshold=1):
    """Compares two words for similarity using WordNet path similarity.

    Args:
        word (str): The first word to compare.
        compare_word (str): The second word to compare with the first.
        threshold (float, optional): The similarity threshold required to consider the words matching.
            Defaults to 1.

    Returns:
        bool: True if the maximum path similarity between any synsets of the two words
        is greater than or equal to the threshold, otherwise False.
    """
    syns1 = wordnet.synsets(word.replace(" ", "_"))
    syns2 = wordnet.synsets(compare_word.replace(" ", "_"))

    max_sim = 0

    for syn1 in syns1:
        for syn2 in syns2:
            sim = syn1.path_similarity(syn2)
            if sim and sim > max_sim:
                max_sim = sim
            if max_sim == 1:
                break

    return max_sim >= threshold


class ObjectRecognitionMetric(ClassicMetric):
    """Implements a simple metric for object detection.

    If any part of the target text (including one of the multiple words) appears in the answer text,
    the answer is considered correct.
    """

    def __evaluate__(self, answer_text, target_text, is_valid):
        """Evaluates the answer text against the target text for object recognition.

        Args:
            answer_text (str): The text provided as the answer.
            target_text (str or list): The target text or list of target strings.
            is_valid (bool): Indicates if the sample is valid for evaluation.

        Returns:
            str: "correct" if recognized, "incorrect" otherwise, or "none" if invalid.
        """
        if not is_valid:
            return "none"

        # Some common synonyms
        # TODO change this to use wordnet substitutions as in CocoObjectDetectionMetric
        answer_text = answer_text.lower()
        answer_text = answer_text.replace("sofa", "couch")
        answer_text = answer_text.replace("sneakers", "shoes")
        answer_text = answer_text.replace("sneaker", "shoes")
        answer_text = answer_text.replace("shoe", "shoes")
        answer_text = answer_text.replace("weight", "dumbbell")

        target_text = target_text.lower()
        try:
            target_text = ast.literal_eval(target_text)
        except:
            pass

        # pair case has 2 targets, single has one
        if isinstance(target_text, list):
            a = target_text[0]
            b = target_text[1]
        else:
            a = target_text
            b = None

        # search over parts of a
        a_parts = a.split(" ")
        ps = []
        for p in a_parts:
            correct = True if p in answer_text else False
            ps.append(correct)
        pred_a_bool = any(ps)

        if b:
            # search over parts of b
            b_parts = b.split(" ")
            ps = []
            for p in b_parts:
                correct = True if p in answer_text.lower() else False
                ps.append(correct)
            pred_b_bool = any(ps)

            # correct if both correct
            pred = "correct" if (pred_a_bool and pred_b_bool) else "incorrect"
        else:
            pred = "correct" if pred_a_bool else "incorrect"

        return pred


class CocoObjectDetectionMetric(DetectionMetric):
    """Implements parsing to prepare for COCO detection metrics.

    The model output is parsed and formed into a COCO annotation, which is returned and stored as the
    metric output. The final statistics are computed in the CocoDetectionAggregator.
    """

    def __init__(self, target_coco_json_reader: JsonReader):
        """Initializes the metric with the ground-truth COCO JSON data.

        Args:
            target_coco_json_reader (JsonReader): Reader to load the ground truth JSON for
                the detections (in COCO JSON format).
        """
        super().__init__()

        # create COCO class and populate with groundtruth json data
        self.coco = COCO()
        self.coco.dataset = target_coco_json_reader.read()
        self.coco.createIndex()

        # get a list of all categories
        coco_cat_ids = self.coco.getCatIds()
        coco_cats = self.coco.loadCats(coco_cat_ids)
        self.coco_cat_name_to_id = {}

        # create a dict to look up category ID by name
        for c in coco_cats:
            self.coco_cat_name_to_id[c["name"]] = c["id"]

    def __evaluate__(self, image_id, answer_text, is_valid):
        """Evaluates the detections extracted from the answer text for a given image.

        Args:
            image_id (int): The identifier of the image being evaluated.
            answer_text (str): The text output from the model, containing detections.
            is_valid (bool): Indicates if the sample is valid for evaluation.

        Returns:
            str: A JSON-formatted string representation of the COCO-style annotations.
        """
        if not is_valid:
            return "none"

        # load image info, need width and height
        img = self.coco.loadImgs(image_id)
        w = img[0]["width"]
        h = img[0]["height"]

        # split each line into a separate detection
        answer_text = answer_text.strip()
        dets = answer_text.split("\n")

        annotations = []

        for det in dets:
            try:
                # parse the detection format (as specified in the prompt)
                parts = det.split("-")
                assert len(parts) == 3, f"Error parsing detection: {det}"
                box_string, label, confidence = parts
                confidence = float(confidence.strip())

                # form in COCO style
                box = ast.literal_eval(box_string.strip())
                xmin, ymin, xmax, ymax = box
                box = [xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h]

                label = label.strip()

                # store the detection
                # use wordnet distance to handle synonyms
                for cat in self.coco_cat_name_to_id.keys():
                    if wordnet_compare(label, cat):
                        annotation = {
                            "image_id": image_id,
                            "category_id": self.coco_cat_name_to_id[cat],
                            "bbox": box,
                            "score": confidence,
                        }
                        annotations.append(annotation)
                        break

            except Exception as e:
                logging.error(f"Error parsing detection: {e} answer_text:{answer_text}")
                import traceback

                traceback.print_exc()

        return str(annotations)
