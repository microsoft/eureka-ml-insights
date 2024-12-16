import ast

from .metrics_base import ExactMatch, Metric
import xml.etree.ElementTree as ET

import re

class NPHardMetricSAT(Metric):
    """This class is a metric that requires a correct prediction to be only one of the valid multiple choice answers."""

    def __init__(self):
        super().__init__()


    def extract_solution(self, model_output: str):
        # Regex to match something like:
        # **Solution: (True, True, True, True)**
        pattern = r"\*?\*?Solution:\s*\((.*?)\)\*?\*?"
        match = re.search(pattern, model_output)
        if not match:
            return None
        
        # Extract the inner content of the parentheses
        content = match.group(1)

        # Now parse the tuple content by splitting on commas and stripping spaces
        # Example: "True, True, True, True" -> [True, True, True, True]
        values = [v.strip() for v in content.split(',')]
        # Convert each string in values to actual booleans if possible
        bool_values = []
        for v in values:
            if v.lower() == 'true':
                bool_values.append(True)
            elif v.lower() == 'false':
                bool_values.append(False)
            else:
                # If there's something non-boolean, return None or handle differently
                return None
        
        return tuple(bool_values)


    def solution_in_ground_truth(self, solution, ground_truth):

        ### if solution is None ###

        if not solution:
            return False
        
        if not ground_truth:
            # If no solution was found by the model (i.e., solution is None or indicates unsatisfiable),
            # it matches the ground truth unsatisfiable scenario.
            # Here we assume `None` for no solution means unsatisfiable.
            if ("unsatisfiable" in solution):
                return True
            else:
                return False
        
        # Convert (True, True, True, True) to (1, 1, 1, 1)
        int_solution = tuple(int(v) for v in solution)
        
        # Convert tuple to string format like "(1, 1, 1, 1)"
        solution_str = "(" + ", ".join(str(v) for v in int_solution) + ")"
        
        # Check if solution_str is in the ground_truth list
        return solution_str in ground_truth


    def __evaluate__(self, x):
        is_valid_curr=x["is_valid"]

        if not is_valid_curr:
            return "none"
        
        # print("came to evaluate\n")

        model_output_curr=x["model_output"]
        ground_truth_curr=x["ground_truth"]
        
        solution_curr = self.extract_solution(model_output_curr)
        print("\n")
        print("Solution: ", solution_curr)
        print("Ground truth: ", ground_truth_curr)

        final_answer_curr = self.solution_in_ground_truth(solution_curr, ground_truth_curr)


        print("Final answer: ", final_answer_curr)

        return "correct"

