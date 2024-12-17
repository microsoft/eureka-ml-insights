import ast

from .metrics_base import ExactMatch, Metric
import xml.etree.ElementTree as ET


class NPHardMetric(Metric):
# class NPHardMetric():
    """This class is a metric that requires a correct prediction to be only one of the valid multiple choice answers."""

    def __init__(self):
        super().__init__()

    def parse_xml_to_dict(self, xml_string):
        # Parse the XML string
        root = ET.fromstring(xml_string)

        # Find the 'final_answer' tag
        final_answer_element = root.find('final_answer')

        # Find the 'reasoning' tag
        reasoning_element = root.find('reasoning')

        return final_answer_element, reasoning_element

    def __is_valid_tsp_path(self, path, cities, distance_matrix=None):
        """
        Validates a TSP path and optionally evaluates its length.

        Parameters:
            path (list): The TSP path, a list of city indices (or names).
            cities (list): The list of all cities.
            distance_matrix (list of lists, optional): Matrix representing distances between cities.

        Returns:
            tuple: (bool, float): Whether the path is valid and its length (if valid). Length is None if invalid.
        """
        # Ensure the path is not empty and has the correct number of cities
        if not path or len(path) != len(cities) + 1:
            print("Invalid: Path is empty or has incorrect number of nodes.")
            return False, None

        # Ensure the path starts and ends at the same city
        if path[0] != path[-1]:
            print("Invalid: Path does not start and end at the same city.")
            return False, None

        # Ensure all cities are visited exactly once (except the start/end city)
        unique_cities_in_path = set(path[:-1])  # Exclude the last city
        unique_cities = set(cities)

        if unique_cities_in_path != unique_cities:
            print("Invalid: Path does not include all cities exactly once.")
            return False, None

        # Ensure there are no duplicate visits to cities (except for the first/last city)
        if len(path[:-1]) != len(unique_cities_in_path):
            print("Invalid: Path includes duplicate visits to cities.")
            return False, None

        # If a distance matrix is provided, calculate the path length
        path_length = 0
        if distance_matrix:
            try:
                for i in range(len(path) - 1):
                    start = cities.index(path[i])
                    end = cities.index(path[i + 1])
                    path_length += distance_matrix[start][end]
            except (IndexError, ValueError):
                print("Invalid: Path contains cities not in the provided distance matrix.")
                return False, None

        print("Valid TSP path.")
        return True, path_length


    def __evaluate__(self, x):
        is_valid_curr=x["is_valid"]

        if not is_valid_curr:
            return "none"
        
        optimal_tour_curr=x["optimal_tour"]
        weight_matrix_curr=x["weight_matrix"]
        ground_truth_curr=x["ground_truth"]
        model_output_curr=x["model_output"]


        final_answer_element, reasoning_element = self.parse_xml_to_dict(model_output_curr)
        tour_distance = ast.literal_eval(final_answer_element.text)['TotalDistance']

        tour_string = ast.literal_eval(final_answer_element.text)['Path']
        tour = list(map(int, tour_string.split('->')))

        print("final tour sring: ", tour_string)
        print("final tour: ", tour)

        cities = [str(i) for i in range(len(weight_matrix_curr))]

        is_valid, total_tsp_path_length = self.__is_valid_tsp_path(tour, cities, weight_matrix_curr)
        print(is_valid, total_tsp_path_length)

        ### incorrect if path is invalid or total_path_length is not same as ground truth path length

        if not is_valid:
            return "incorrect"
        
        if total_tsp_path_length != ground_truth_curr:
            return "incorrect"

        return "correct"





    # def __evaluate__(self, x):
    #     is_valid_curr=x["is_valid"]

    #     if not is_valid_curr:
    #         return "none"
        
    #     optimal_tour_curr=x["optimal_tour"]
    #     weight_matrix_curr=x["weight_matrix"]
    #     ground_truth_curr=x["ground_truth"]
    #     model_output_curr=x["model_output"]


    #     final_answer_element, reasoning_element = self.parse_xml_to_dict(answer_text)
    #     # tour_distance = ast.literal_eval(final_answer_element.text)['TotalDistance']

    #     # tour_string = ast.literal_eval(final_answer_element.text)['Path']
    #     # tour = list(map(int, tour_string.split('->')))

    #     # print("final tour sring: ", tour_string)
    #     # print("final tour: ", tour)



    #     # # Example usage
    #     # cities = ["A", "B", "C", "D"]
    #     # distance_matrix = [
    #     #     [0, 10, 15, 20],  # Distances from A
    #     #     [10, 0, 35, 25],  # Distances from B
    #     #     [15, 35, 0, 30],  # Distances from C
    #     #     [20, 25, 30, 0],  # Distances from D
    #     # ]
    #     # valid_path = ["A", "B", "C", "D", "A"]
    #     # invalid_path = ["A", "B", "C", "C", "A"]

    #     # is_valid, total_tsp_path_length = self.__is_valid_tsp_path(valid_path, cities, distance_matrix)
    #     # print(is_valid, total_tsp_path_length)

    #     # is_valid, total_tsp_path_length = self.__is_valid_tsp_path(invalid_path, cities, distance_matrix)
    #     # print(is_valid, total_tsp_path_length)

    #     # tour=[0,1,2,0]
    #     # distance_matrix=[0,1,2]
    #     # successful_tsp="incorrect"
    #     # # Check if tour is a cycle
    #     # if tour[0] != tour[-1]:            
    #     #     return "incorrect"
        
    #     # # Check if all cities are visited
    #     # if len(tour) != len(distance_matrix) + 1:
    #     #     return "incorrect"
        
    #     # # Check if all cities are visited only once
    #     # if len(tour) != len(distance_matrix) + 1:
    #     #     return "incorrect"

    #     return "correct"
    #     # return super().__evaluate__(tour_distance, target_text, is_valid)
