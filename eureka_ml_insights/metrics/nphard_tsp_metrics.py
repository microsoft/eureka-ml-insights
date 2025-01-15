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

    def __is_tour_present(optimal_tour_curr, tour_string):
        # Remove the enclosing characters and convert the optimal tour to a list of tuples
        optimal_tour_list = eval(optimal_tour_curr.strip('.'))

        # Convert each tuple in optimal_tour_list to a comma-separated string
        optimal_tour_strings = [','.join(map(str, tour)) for tour in optimal_tour_list]

        # Check if tour_string is in the list of optimal tour strings
        return tour_string in optimal_tour_strings


    def __evaluate__(self, x):
        is_valid_curr=x["is_valid"]

        if not is_valid_curr:
            return "none"
        
        # breakpoint()

        optimal_tour_curr=x["optimal_tour"]
        weight_matrix_curr=x["weight_matrix"]
        ground_truth_curr=x["ground_truth"]
        tour_string=x["model_output"]

        tour = list(map(int, tour_string.split(',')))

        print("final tour: ", tour)

        cities = [i for i in range(len(weight_matrix_curr))]

        is_valid, total_tsp_path_length = self.__is_valid_tsp_path(tour, cities, weight_matrix_curr)
        print(is_valid, total_tsp_path_length, ground_truth_curr)

        ### incorrect if path is invalid or total_path_length is not same as ground truth path length

        if not is_valid:
            return "incorrect"
        
        if total_tsp_path_length != ground_truth_curr:
            return "incorrect"

        is_tour_present = self.__is_tour_present(optimal_tour_curr, tour_string)

        if not is_tour_present:
            return "incorrect"

        return "correct"
