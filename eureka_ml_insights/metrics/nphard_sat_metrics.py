import logging

from .metrics_base import Metric, CompositeMetric

class NPHardSATMetric(CompositeMetric):
    """
    A metric class for evaluating solutions to the SAT.
    A prediction is considered correct if it is a valid SAT tour and matches one of the optimal solutions.
    """

    def __init__(self):
        super().__init__()

    def is_sat_soln_present(self, optimal_tour_curr, tour_string):
        
        if tour_string == "" and not optimal_tour_curr:
            return True
        
        optimal_tour_strings = [",".join(item.replace(" ", "").strip("()").split(",")) for item in optimal_tour_curr] #[",".join(item.strip("()").split()) for item in optimal_tour_curr]

        # Check if tour_string is in the list of optimal tour strings
        return tour_string in optimal_tour_strings
    

    def parse_parenthesized_string(self, s):
        """
        Given a string like '(0, 1, 1, 0)', strip parentheses and spaces, 
        then return a list of integers [0, 1, 1, 0].
        """
        # Remove leading/trailing whitespace and parentheses
        s = s.strip().strip('()')
        # Split by comma
        parts = [x.strip() for x in s.split(',')]
        # Convert to integers
        return [int(p) for p in parts if p]
    
    def hamming_distance(self, list1, list2):
        """
        Compute the Hamming distance between two lists of 0/1 integers.
        If the lists differ in length, compare up to the shorter length 
        (you can change this behavior if needed).
        """

        min_len = min(len(list1), len(list2))
        return sum(a != b for a, b in zip(list1[:min_len], list2[:min_len]))

    def find_min_distance(self, reference_list, candidate_strings):
        """
        Given a reference list of 0/1 ints and a list of parenthesized candidate strings,
        compute and return the minimum Hamming distance. If the candidate list is empty,
        return 7 (based on the problem requirement).
        """
        # print(reference_list)
        # print(candidate_strings)
        # print("\n")

        # If there are no candidates, return 7 (per the requirement).
        if not candidate_strings:
            return len(reference_list) # or len(reference_list) if that always equals 7 for your reference
        
        if not reference_list:
            print(candidate_strings[0])
            tuple_str = candidate_strings[0]
            tuple_str = tuple_str.strip("() ").replace(" ", "")
            parts = tuple_str.split(",")
            my_list = [int(x) for x in parts]
            min_distance = my_list.count(0)+my_list.count(1)
            return min_distance
            

        distances = []
        for candidate_str in candidate_strings:
            candidate_list = self.parse_parenthesized_string(candidate_str)
            dist = self.hamming_distance(reference_list, candidate_list)
            distances.append(dist)

        return min(distances)

    def cal_min_hamming_distance(self, optimal_tour_curr, tour_string):
        """
        Compute the Hamming distance between two lists of 0/1 integers.
        If the lists differ in length, compare up to the shorter length 
        (you can change this behavior if needed).
        """
        reference_list = [int(x) for x in tour_string.split(',') if x.strip()]

        min_distance = self.find_min_distance(reference_list, optimal_tour_curr)

        print(reference_list)
        print(optimal_tour_curr)
        print(min_distance)
        print("\n")

        return min_distance

        # distances = []
        
        # for candidate_str in optimal_tour_curr:
        #     candidate_list = self.parse_parenthesized_string(candidate_str)
        #     dist = self.hamming_distance(reference_list, candidate_list)
        #     distances.append(dist)

        # if not distances:
        #     return 0
        
        # return min(distances)


    def __evaluate__(self, x):
        """
        Evaluates whether the model's output is a correct SAT tour.
        """
        is_valid_curr = x["is_valid"]

        if not is_valid_curr:
            return "none"

        optimal_tour_curr = x["solution"]
        tour_string = x["model_output"]

        # Check if the predicted tour is one of the optimal solutions
        is_SAT_tour_present = self.is_sat_soln_present(optimal_tour_curr, tour_string)

        # breakpoint()

        # print(optimal_tour_curr)
        # print(tour_string)        

        dist_path_gt=self.cal_min_hamming_distance(optimal_tour_curr, tour_string)

        if not is_SAT_tour_present:
            return {"results": "incorrect", "dist_results": dist_path_gt}

        return {"results": "correct", "dist_results": dist_path_gt}
