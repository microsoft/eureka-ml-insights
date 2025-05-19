from .metrics_base import Metric


class NPHardSATMetric(Metric):
    """
    A metric class for evaluating solutions to the SAT.
    A prediction is considered correct if it is a valid SAT assignment and matches one of the optimal solutions.
    """

    def __init__(self):
        super().__init__()

    def is_sat_soln_present(self, optimal_assignment_list, assignment_string):
        """
        Decide whether the model-predicted assignment string is accepted as a correct
        SAT solution.

        1. **Input formats**
        • `optimal_assignment_list` – `List[str]`
            A list of canonical optimal assignments, each as a comma-separated
            string of 0/1 literals in the agreed variable orde, e.g.
            `"1,0,1,1"`.
            – An **empty list (`[]`) represents “unsatisfiable.”**
        • `assignment_string` – `str`
            The model’s prediction in the *same* 0/1 comma-separated format.
            – An empty string (`""`) means the model declares “unsatisfiable.”

        2. **Normalisation performed**
        The function removes spaces and surrounding parentheses from every entry
        in `optimal_assignment_list`, then rejoins tokens with single commas,
        e.g. `"(1, 0 ,1)" → "1,0,1"`.

        3. **Acceptance criteria**
        • Return **`True`** if
            a. *Both* sides claim unsatisfiable
                (`assignment_string == ""` **and** `optimal_assignment_list == []`), **or**
            b. The canonical `assignment_string` exactly matches (string equality)
                **one** element of the normalised `optimal_assignment_list`.
        • Otherwise return **`False`**.

        4. **Order sensitivity**
        Because matching is string-exact, **variable order must match** between
        prediction and ground truth.

        Returns
        -------
        bool
            `True` if the predicted assignment is accepted as correct, else `False`.
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
        """
        Evaluates whether the model's output is a correct SAT assignment.
        """
        is_valid_curr = x["is_valid"]

        if not is_valid_curr:
            return "none"

        optimal_assignment_list = x["solution"]
        assignment_string = x["extracted_answer"]

        # Check if the predicted assignment is one of the optimal solutions
        is_assignment_optimal = self.is_sat_soln_present(optimal_assignment_list, assignment_string)

        if not is_assignment_optimal:
            return "incorrect"

        return "correct"
