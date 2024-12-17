# write unit tests for the classes in data_utils/transform.py

import logging
import unittest

import pandas as pd

from eureka_ml_insights.data_utils.spatial_utils import extract_answer_from_text_map_and_maze

log = logging.getLogger("VISION_LANGUAGE_extract_answer_from_text_map_and_maze_tests")


class TestExtractAnswer(unittest.TestCase):
    def test_extract_answer_from_text_map_and_maze(self):

        ## Direction answer extraction tests
        testcases_dir = [
            "The correct answer is A: Planetarium Prints is located Northeast of Police Supply Store.",

            "Planetarium Prints is located to the Northeast of Police Supply Store, so the answer is A.\n\nTherefore, A (Northeast) is the correct answer as it accurately describes the position of Planetarium Prints relative to Police Supply Store.",

            "A. Northeast\n\nBased on the information provided, Planetarium Prints is located to the Northeast of Police Supply Store. This can be determined through the following steps:\n\n1. Police Supply Store is used as the reference point.\n2. Coral Crafts is stated to be Northwest of Police Supply Store.\n3. Planetarium Prints is then described as being Southeast of Coral Crafts.\n4. Since Coral Crafts is Northwest of Police Supply Store, and Planetarium Prints is Southeast of Coral Crafts, this places Planetarium Prints to the Northeast of Police Supply Store.\n\nAdditionally, the description directly states \"Planetarium Prints is to the Northeast of Police Supply Store\", confirming the answer.",

            "Planetarium Prints is **Northeast** of Police Supply Store. \n\nHere's the step-by-step explanation:\n\n1. **Establish a reference point:** The question asks for the direction of Planetarium Prints *relative* to Police Supply Store.  This means we consider Police Supply Store as our central point.\n\n2. **Use the given information:** We are told \"Planetarium Prints is to the Northeast of Police Supply Store.\"\n\n3. **Direct relationship:** Since we have a direct statement about the relative position of Planetarium Prints to Police Supply Store, we don't need to analyze the other clues.\n\n4. **Final answer:** Therefore, Planetarium Prints is located **Northeast** of Police Supply Store. \n",

            "Planetarium Prints is Northeast of Police Supply Store.\n\n**Reasoning:**\n\n1. **Identify the relative positions:**\n   - Police Supply Store is the reference point.\n   - Planetarium Prints is one of the objects we need to locate relative to the Police Supply Store.\n\n2. **Analyze the given directions:**\n   - Planetarium Prints is described as being to the Northeast of Police Supply Store.\n\n3. **Verify with the map:**\n   - On the map, Planetarium Prints is positioned to the upper right (Northeast) of the Police Supply Store.\n\n4. **Conclusion:**\n   - Based on the description and the map, Planetarium Prints is indeed Northeast of Police Supply Store.\n\nThus, the correct answer is:\nA. Northeast",

            "Planetarium Prints is to the **Northeast** of Police Supply Store (Option A).\n\n - Based on the above placements, **Planetarium Prints is to the Northeast of Police Supply Store**.\n\nTherefore, the correct answer is **Option A: Northeast**.",

            "The correct answer is A. Northeast.\n\nTo determine the direction of Planetarium Prints relative to Police Supply Store, we can follow the given descriptions:\n\n1. Police Supply Store is in the map.\n2. Narwhal's Novelties is to the Northwest of Police Supply Store.\n3. Coral Crafts is to the Northwest of Narwhal's Novelties.\n4. Coral Crafts is to the Northwest of Police Supply Store.\n5. Planetarium Prints is to the Southeast of Coral Crafts.\n6. Planetarium Prints is to the Northeast of Police Supply Store.\n7. Oz Oddities is to the Southwest of Planetarium Prints.\n8. Oz Oddities is to the Southwest of Police Supply Store.\n9. Ice Queen Ice Cream is to the Northwest of Planetarium Prints.\n10. Ice Queen Ice Cream is to the Southeast of Coral Crafts.\n\nFrom these descriptions, we can see that Planetarium Prints is located to the Northeast of Police Supply Store. This is directly stated in point 6. Therefore, the correct answer is A. Northeast.",

            "A. Northeast.\n\nPlanetarium Prints is to the Northeast of Police Supply Store. The reasoning behind this answer is based on the provided directions and locations of the objects on the map. Narwhal's Novelties is to the Northwest of Police Supply Store, and Coral Crafts is to the Northwest of Narwhal's Novelties. Planetarium Prints is to the Southeast of Coral Crafts, which means it is also to the Northeast of Police Supply Store.",
        ]
        
        target_options_dir = [["A. Northeast","B. Northwest","C. Southwest","D. Southeast."]] * len(testcases_dir)
        correct_answers_dir = ["northeast"] * len(testcases_dir)

        ## Numerical extraction tests

        testcases_numerical = [
            "A. 1\n\nTo determine how many objects are in the Southeast of Oz Oddities, we need to look at the relative positions of the objects on the map:\n\n1. Oz Oddities is located at the bottom of the map.\n2. Directly to the Northeast of Oz Oddities is the Police Supply Store.\n3. To the Southeast of Oz Oddities, there is only one object, which is Planetarium Prints.\n4. All other objects are either to the North or Northwest of Oz Oddities and therefore not in the Southeast direction.\n\nBased on the map, only Planetarium Prints is in the Southeast of Oz Oddities, which means the correct answer is A. 1.",

            "There are zero objects",

            "There are no objects",
        ]

        target_options_numerical= [["A. 1","B. 0","C. 2","D. 3."]] * len(testcases_numerical)
        correct_answers_numerical = ["1", "0", "0"]

        target_options = target_options_dir + target_options_numerical
        testcases = testcases_dir + testcases_numerical
        correct_answers = correct_answers_dir + correct_answers_numerical

        results = []
        for i, test in enumerate(testcases):
            extracted_answer = extract_answer_from_text_map_and_maze(test, target_options[i])
            results.append(correct_answers[i].lower() in extracted_answer.lower())

        self.assertTrue(all(results))

if __name__ == "__main__":
    unittest.main()
