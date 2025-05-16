from .metrics_base import Metric


class NPHardSATMetric(Metric):
    """
    A metric class for evaluating solutions to the SAT.
    A prediction is considered correct if it is a valid SAT tour and matches one of the optimal solutions.
    """

    def __init__(self):
        super().__init__()

    def is_sat_soln_present(self, optimal_tour_curr, tour_string):
        # If both the model output and the ground truth say the solution is unsatisfiable,
        # (tour_string: "", ground truth: empty list), accept immediately.
        if tour_string == "" and not optimal_tour_curr:
            return True

        optimal_tour_strings = [",".join(item.replace(" ", "").strip("()").split(",")) for item in optimal_tour_curr]

        # Check if tour_string is in the list of optimal tour strings
        return tour_string in optimal_tour_strings

    def __evaluate__(self, x):
        """
        Evaluates whether the model's output is a correct SAT tour.
        """
        is_valid_curr = x["is_valid"]

        if not is_valid_curr:
            return "none"

        optimal_tour_curr = x["solution"]
        tour_string = x["extracted_answer"]

        # Check if the predicted tour is one of the optimal solutions
        is_SAT_tour_present = self.is_sat_soln_present(optimal_tour_curr, tour_string)

        if not is_SAT_tour_present:
            return "incorrect"

        return "correct"
