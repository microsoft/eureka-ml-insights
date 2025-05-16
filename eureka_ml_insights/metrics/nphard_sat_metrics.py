from .metrics_base import Metric


class NPHardSATMetric(Metric):
    """
    A metric class for evaluating solutions to the SAT.
    A prediction is considered correct if it is a valid SAT assignment and matches one of the optimal solutions.
    """

    def __init__(self):
        super().__init__()

    def is_sat_soln_present(self, optimal_assignment_curr, assignment_string):
        # If both the model output and the ground truth say the solution is unsatisfiable,
        # (assignment_string: "", ground truth: empty list), accept immediately.
        if assignment_string == "" and not optimal_assignment_curr:
            return True

        optimal_assignment_strings = [",".join(item.replace(" ", "").strip("()").split(",")) for item in optimal_assignment_curr]

        # Check if assignment_string is in the list of optimal assignment strings
        return assignment_string in optimal_assignment_strings

    def __evaluate__(self, x):
        """
        Evaluates whether the model's output is a correct SAT assignment.
        """
        is_valid_curr = x["is_valid"]

        if not is_valid_curr:
            return "none"

        optimal_assignment_curr = x["solution"]
        assignment_string = x["extracted_answer"]

        # Check if the predicted assignment is one of the optimal solutions
        is_SAT_assignment_present = self.is_sat_soln_present(optimal_assignment_curr, assignment_string)

        if not is_SAT_assignment_present:
            return "incorrect"

        return "correct"
