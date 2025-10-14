from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    multi_turn_checker,
)

from eureka_ml_insights.metrics.metrics_base import ClassicMetric


class BFCLMultiturnMatch(ClassicMetric):
    """This metric class checks if the generated function calls match the ground-truth."""

    def __init__(
        self,
        model_output_col: str = "model_output",
        ground_truth_col: str = "ground_truth",
        initial_config_col: str = "initial_config",
        involved_classes_col: str = "involved_classes",
        test_entry_id_col: str = "id",
    ):
        super().__init__()
        self.model_output_col = model_output_col
        self.ground_truth_col = ground_truth_col
        self.initial_config_col = initial_config_col
        self.involved_classes_col = involved_classes_col
        self.test_entry_id_col = test_entry_id_col

    def __evaluate__(self, answer_text, target_text, initial_config, involved_classes, test_entry_id):
        test_entry = {
            "initial_config": eval(initial_config),
            "involved_classes": eval(involved_classes),
            "id": test_entry_id,
        }
        test_category = ""  # following the bfcl_eval's original eval code.
        model_name = ""  # following the bfcl_eval's original eval code.
        multi_turn_ground_truth_list = eval(target_text)

        accuracy_checker_result = multi_turn_checker(
            answer_text,
            multi_turn_ground_truth_list,
            test_entry,
            test_category,
            model_name,
        )
        return str(accuracy_checker_result["valid"])

    def evaluate(self, data):
        self.validate_data(data)
        data[self.__class__.__name__ + "_result"] = data.apply(
            lambda x: self.__evaluate__(
                x[self.model_output_col],
                x[self.ground_truth_col],
                x[self.initial_config_col],
                x[self.involved_classes_col],
                x[self.test_entry_id_col],
            ),
            axis=1,
        )
        return data
