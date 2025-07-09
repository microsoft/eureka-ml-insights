"""This module provides the NPHardSATMetric class for evaluating solutions to the SAT problem.

The NPHardSATMetric class inherits from Metric and determines whether a model-predicted
assignment string is one of the optimal solutions, or whether both the prediction and 
ground truth indicate the problem is unsatisfiable.
"""

from .metrics_base import Metric


class NPHardSATMetric(Metric):
    """NPHardSATMetric class for evaluating solutions to SAT problems.

    A prediction is considered correct if it is a valid SAT assignment that
    matches one of the optimal solutions, or if both the model output and ground
    truth indicate unsatisfiable.
    """

    def __init__(self):
        """Initializes the NPHardSATMetric class."""
        super().__init__()

    def is_assignment_optimal(self, optimal_assignment_list, assignment_string):
        """Determines whether a model-predicted assignment string is accepted as a correct SAT solution.

        Args:
            optimal_assignment_list (List[str]): A list of canonical optimal assignments,
                each expressed as a comma-separated string of 0/1 literals, following the
                variable order specified in the prompt instructions (e.g. "1,0,1,1"). An
                empty list ([]) represents “unsatisfiable.”
            assignment_string (str): The model's prediction in the same 0/1 comma-separated
                format. An empty string ("") means the model declares “unsatisfiable.”

        Normalization:
            This function removes spaces and surrounding parentheses from every entry in
            optimal_assignment_list, then rejoins tokens with single commas.
            For example, "(1, 0 ,1)" becomes "1,0,1".

        Acceptance Criteria:
            1. Returns True if both sides claim unsatisfiable
               (assignment_string == "" and optimal_assignment_list == []).
            2. Returns True if the canonical assignment_string exactly matches one element
               of the normalized optimal_assignment_list.
            Otherwise, returns False.

        Order Sensitivity:
            Because matching is string-exact, variable order must match between
            prediction and ground truth.

        Returns:
            bool: True if the predicted assignment is accepted as correct, otherwise False.
        """
        # If both the model output and the ground truth say the solution is unsatisfiable,
        # (assignment_string: "", ground truth: empty list), accept immediately.
        if assignment_string == "" and not optimal_assignment_list:
            return True

        optimal_assignment_list = [
            ",".join(item.replace(" ", "").strip("()").split(",")) for item in optimal_assignment_list
        ]

        # Check if assignment_string is in the list of optimal assignment strings
        return assignment_string in optimal_assignment_list

    def __evaluate__(self, x):
        """Evaluates whether the model's output is a correct SAT assignment.

        Args:
            x (dict): A dictionary containing the model's prediction and ground truth
                information. It should include:
                - "is_valid" (bool): Whether the model's output is valid.
                - "solution" (List[str]): The list of optimal solutions.
                - "extracted_answer" (str): The model's predicted assignment string.

        Returns:
            str:
                - "none" if the prediction is invalid.
                - "incorrect" if the prediction does not match an optimal solution.
                - "correct" if the prediction matches an optimal solution.
        """
        is_valid_curr = x["is_valid"]

        if not is_valid_curr:
            return "none"

        optimal_assignment_list = x["solution"]
        assignment_string = x["extracted_answer"]

        # Check if the predicted assignment is one of the optimal solutions
        is_assignment_optimal = self.is_assignment_optimal(optimal_assignment_list, assignment_string)

        if not is_assignment_optimal:
            return "incorrect"

        return "correct"
