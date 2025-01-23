import logging

from .metrics_base import Metric


class NPHardTSPMetric(Metric):
    """
    A metric class for evaluating solutions to the Traveling Salesman Problem (TSP).
    A prediction is considered correct if it is a valid TSP tour and matches one of the optimal solutions.
    """

    def __init__(self):
        super().__init__()

    def is_valid_tsp_path(self, path, cities, distance_matrix=None):
        """
        Validates a TSP path and evaluates its length.

        Parameters:
            path (list): The TSP path, a list of city indices.
            cities (list): The list of all cities.
            distance_matrix (list of lists, optional): Matrix representing distances between cities.

        Returns:
            tuple: (bool, float): Whether the path is valid and its length (if valid). Length is None if invalid.
        """
        # Ensure the path is not empty and has the correct number of cities
        if not path or len(path) != len(cities) + 1:
            logging.info("Invalid: Path is empty or has incorrect number of nodes.")
            return False, None

        # Ensure the path starts and ends at the same city
        if path[0] != path[-1]:
            logging.info("Invalid: Path does not start and end at the same city.")
            return False, None

        # Ensure all cities are visited exactly once (except the start/end city)
        unique_cities_in_path = set(path[:-1])  # Exclude the last city
        unique_cities = set(cities)

        if unique_cities_in_path != unique_cities:
            logging.info("Invalid: Path does not include all cities exactly once.")
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
                logging.info("Invalid: Path contains cities not in the provided distance matrix.")
                return False, None

        return True, path_length

    def is_tour_present(self, optimal_tour_curr, tour_string):

        optimal_tour_list = eval(optimal_tour_curr.strip("."))
        optimal_tour_strings = [",".join(map(str, tour + (tour[0],))) for tour in optimal_tour_list]

        # Check if tour_string is in the list of optimal tour strings
        return tour_string in optimal_tour_strings

    def __evaluate__(self, x):
        """
        Evaluates whether the model's output is a correct TSP tour.
        """
        is_valid_curr = x["is_valid"]

        if not is_valid_curr:
            return "none"

        optimal_tour_curr = x["optimal_tour"]
        weight_matrix_curr = x["weight_matrix"]
        ground_truth_curr = x["ground_truth"]
        tour_string = x["model_output"]

        # Convert tour string into a list of integers representing the city sequence
        tour = list(map(int, tour_string.split(",")))
        cities = [i for i in range(len(weight_matrix_curr))]

        # Validate the TSP tour and compute its length
        is_tsp_path_valid, total_tsp_path_length = self.is_valid_tsp_path(tour, cities, weight_matrix_curr)

        # The prediction is incorrect if the tour is invalid or the length is incorrect
        if not is_tsp_path_valid or total_tsp_path_length != ground_truth_curr:
            return "incorrect"

        # Check if the predicted tour is one of the optimal solutions
        is_tsp_tour_present = self.is_tour_present(optimal_tour_curr, tour_string)

        if not is_tsp_tour_present:
            return "incorrect"

        return "correct"
