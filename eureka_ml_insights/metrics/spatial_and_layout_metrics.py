import ast
import logging
import re

import nltk
from pycocotools.coco import COCO

from eureka_ml_insights.data_utils import JsonReader
from eureka_ml_insights.metrics.metrics_base import (
    ClassicMetric,
    DetectionMetric,
    MultipleChoiceMetric,
)


def download_nltk_resources():
    """Download 'wordnet' if not already installed"""
    try:
        nltk.data.find("corpora/wordnet.zip")
    except LookupError:
        nltk.download("wordnet")


download_nltk_resources()
from nltk.corpus import wordnet


class SpatialAndLayoutReasoningMetric(MultipleChoiceMetric):
    """This class is a metric that requires a correct prediction to be only one of the valid multiple choice answers."""

    def __init__(self):
        super().__init__()

    def __evaluate__(self, answer_text, target_text, target_options, is_valid):
        if not is_valid:
            return "none"

        # count which options appear
        options_count = {}
        for option in target_options:

            # make sure only matches whole words by includeing a word boundary "\b" in the pattern
            pattern = "\\b{phrase}\\b".format(phrase=option)

            matches = re.findall(pattern, answer_text, flags=re.IGNORECASE)  # search
            options_count[option] = len(matches)  # count

        total_count = sum(options_count.values())

        # correct if only the right answer appears once,
        # incorrect if only a wrong answer appears,
        # none if there are mutiple answers or nothing matches
        return (
            "correct"
            if (total_count == 1 and options_count[target_text] == 1)
            else "incorrect" if (total_count == 1) else "none"
        )


def wordnet_compare(word, compare_word, threshold=1):

    syns1 = wordnet.synsets(word.replace(" ", "_"))
    syns2 = wordnet.synsets(compare_word.replace(" ", "_"))

    max_sim = 0

    for syn1 in syns1:
        for syn2 in syns2:
            sim = syn1.path_similarity(syn2)

            if sim > max_sim:
                max_sim = sim

            if max_sim == 1:
                break

    return max_sim >= threshold


class ObjectRecognitionMetric(ClassicMetric):
    """
    This class implements a simple metric for object detection.
    If any part of the target_text (including one of the mutiple words) appears in the answer_text,
    the answer is considered to be correct.
    """

    def __evaluate__(self, answer_text, target_text, is_valid):
        if not is_valid:
            return "none"

        # Some common synonyms
        # TODO change this to use wordnet subsitutions as in CocoObjectDetectionMetric
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
    """
    This class implements parsing to prep for a COCO detection metrics.
    The model output is parsed and formed into a COCO annotation.
    This is then returned and stored as the metric output.
    The stats are computed in the CocoDetectionAggregator.
    """

    def __init__(self, target_coco_json_reader: JsonReader):
        """
        args:
            target_coco_json_reader: JsonReader, reader to load the ground truth json for the detections (in coco json format)
        """
        super().__init__()

        # create COCO class and populate with grountruth json data
        self.coco = COCO()
        self.coco.dataset = target_coco_json_reader.read()
        self.coco.createIndex()

        # get the list of images
        coco_img_ids = self.coco.getImgIds()
        coco_imgs = self.coco.loadImgs(coco_img_ids)
        self.coco_file_name_to_id = {}

        # create a dict to look up image id by filename
        for c in coco_imgs:
            self.coco_file_name_to_id[c["file_name"]] = c["id"]

        # get a lst of all cats
        coco_cat_ids = self.coco.getCatIds()
        coco_cats = self.coco.loadCats(coco_cat_ids)
        self.coco_cat_name_to_id = {}

        # create a dict to look up cat id by
        for c in coco_cats:
            self.coco_cat_name_to_id[c["name"]] = c["id"]

    def __evaluate__(self, images, answer_text, is_valid):
        if not is_valid:
            return "none"

        # load image info, need w and h
        image = images[0]
        image_id = self.coco_file_name_to_id[image]
        img = self.coco.loadImgs(image_id)
        w = img[0]["width"]
        h = img[0]["height"]

        # split each line into a seperate detection
        answer_text = answer_text.strip()
        dets = answer_text.split("\n")

        annotations = []

        for det in dets:
            try:
                # parse the detection format (as specified int he prompt)
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
                            "image_id": self.coco_file_name_to_id[image],
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
