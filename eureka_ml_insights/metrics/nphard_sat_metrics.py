import logging

from .metrics_base import Metric

class NPHardSATMetric(Metric):
    """
    A metric class for evaluating solutions to the SAT.
    A prediction is considered correct if it is a valid SAT tour and matches one of the optimal solutions.
    """

    def __init__(self):
        super().__init__()

    # def is_sat_soln_present(self, optimal_tour_curr, tour_string):

    #     optimal_tour_list = eval(optimal_tour_curr.strip("."))
    #     optimal_tour_strings = [",".join(map(str, tour + (tour[0],))) for tour in optimal_tour_list]

    #     # Check if tour_string is in the list of optimal tour strings
    #     return tour_string in optimal_tour_strings
    
    def is_sat_soln_present(self, optimal_tour_curr, tour_string):
        
        if tour_string == "" and not optimal_tour_curr:
            return True
        
        breakpoint()

        
        # optimal_tour_list = eval(optimal_tour_curr.strip("."))
        optimal_tour_strings = [",".join(map(str, tour + (tour[0],))) for tour in optimal_tour_curr]



        # Check if tour_string is in the list of optimal tour strings
        return tour_string in optimal_tour_strings
    

    def __evaluate__(self, x):
        """
        Evaluates whether the model's output is a correct SAT tour.
        """
        is_valid_curr = x["is_valid"]

        if not is_valid_curr:
            return "none"

        # breakpoint()
        

        # optimal_tour_curr = x["optimal_tour"]
        # weight_matrix_curr = x["weight_matrix"]
        optimal_tour_curr = x["solution"]
        tour_string = x["model_output"]

        # print(optimal_tour_curr)
        # print(tour_string)
        # print("\n")


        # Check if the predicted tour is one of the optimal solutions
        is_SAT_tour_present = self.is_sat_soln_present(optimal_tour_curr, tour_string)

        if not is_SAT_tour_present:
            return "incorrect"

        return "correct"
