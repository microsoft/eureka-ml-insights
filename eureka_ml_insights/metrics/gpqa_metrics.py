import re
from .metrics_base import ClassicMetric

class GPQAMetric(ClassicMetric):
    def validate_data(self, data):
        """This method checks if the data has the required fields."""
        assert "model_output" in data.columns, "Data does not have 'model_output' field."
        assert "ground_truth" in data.columns, "Data does not have 'ground_truth' field."
        return True
    
    def _extract_answer(self, model_output):
        """
        Expect the model output to end with "My answer is ...".
        """
        # Regex pattern to match "My answer is" followed by a letter and either a space or punctuation
        match = re.search(r"My answer is (\w)(?=\s|\W|$)", model_output)
        if match:
            return match.group(1)
        return None

    def __evaluate__(self, model_output, ground_truth):
        letter_answer = self._extract_answer(model_output).upper()
        ground_truth = ground_truth.upper()

        if letter_answer == ground_truth:
            return "correct" 
        return "incorrect"

    def evaluate(self, data):
        self.validate_data(data)
        data[self.__class__.__name__ + "_result"] = data.apply(
            lambda x: self.__evaluate__(x["model_output"], x["ground_truth"]),
            axis=1,
        )
        return data