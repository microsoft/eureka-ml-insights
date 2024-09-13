from tqdm.auto import tqdm


class Metric:
    def validate_data(self, data):
        """This method checks if the data has the required fields."""

        assert "is_valid" in data.columns, "Data does not have 'is_valid' field."
        return True

    def __evaluate__(self, **kwargs):
        raise NotImplementedError

    def evaluate(self, data):
        self.validate_data(data)
        data[self.__class__.__name__ + "_result"] = data.apply(lambda x: self.__evaluate__(x), axis=1)
        return data


class CompositeMetric(Metric):
    """This class is a base class for composite metrics that consist of several submetrics.
    Returns a dictionary mapping "metric_name" to value for each row of data.
    """

    def evaluate(self, data):
        super().evaluate(data)
        data = self.decompose_metric(data)
        return data

    def decompose_metric(self, data):
        composite_metric_col = self.__class__.__name__ + "_result"
        for metric_name in data[composite_metric_col][0]:
            data[self.__class__.__name__ + "_" + metric_name] = data.apply(
                lambda row: (row[composite_metric_col][metric_name]), axis=1
            )
        data.drop(composite_metric_col, axis=1, inplace=True)
        return data


class ClassicMetric(Metric):
    """This class is a base class for metrics that require ground truths and predictions."""

    def validate_data(self, data):
        """This method checks if the data has the required fields."""
        super().validate_data(data)
        assert "model_output" in data.columns, "Data does not have 'model_output' field."
        assert "ground_truth" in data.columns, "Data does not have 'ground_truth' field."
        return True

    def evaluate(self, data):
        self.validate_data(data)
        data[self.__class__.__name__ + "_result"] = data.apply(
            lambda x: self.__evaluate__(x["model_output"], x["ground_truth"], x["is_valid"]), axis=1
        )
        return data


class DetectionMetric(Metric):
    """
    This class is for the detection metric, where access is needed to the image
    and ground truth is stored in an external file.
    """

    def validate_data(self, data):
        """This method checks if the data has the required fields."""

        assert "id" in data.columns, "Data does not have 'id' field."
        assert "model_output" in data.columns, "Data does not have 'model_output' field."
        assert "is_valid" in data.columns, "Data does not have 'is_valid' field."
        return True

    def evaluate(self, data):
        self.validate_data(data)

        tqdm.pandas()

        data[self.__class__.__name__ + "_result"] = data.progress_apply(
            lambda x: self.__evaluate__(x["id"], x["model_output"], x["is_valid"]), axis=1
        )
        return data


class MultipleChoiceMetric(ClassicMetric):
    """This class is a base class for metrics that have a multiple choice answer."""

    def validate_data(self, data):
        """This method checks if the data has the required fields."""
        super().validate_data(data)
        assert "target_options" in data.columns, "Data does not have 'target_options' field."
        return True

    def evaluate(self, data):
        self.validate_data(data)
        data[self.__class__.__name__ + "_result"] = data.apply(
            lambda x: self.__evaluate__(x["model_output"], x["ground_truth"], x["target_options"], x["is_valid"]),
            axis=1,
        )
        return data


class MixedQuestionTypeMetric(MultipleChoiceMetric):
    """This class is a base class for metrics that have a multiple choice or open answer."""

    def validate_data(self, data):
        """This method checks if the data has the required fields."""
        super().validate_data(data)
        assert "question_type" in data.columns, "Data does not have 'question_type' field."
        return True

    def evaluate(self, data):
        self.validate_data(data)
        data[self.__class__.__name__ + "_result"] = data.apply(
            lambda x: self.__evaluate__(
                x["model_output"], x["ground_truth"], x["question_type"], x["target_options"], x["is_valid"]
            ),
            axis=1,
        )
        return data


class SubstringExistsMatch(ClassicMetric):
    """This class checks for a case-insensitive substring match."""

    def __evaluate__(self, answer_text, target_text, is_valid):
        if not is_valid:
            return "none"
        return "correct" if target_text.lower() in answer_text.lower() else "incorrect"


class ExactMatch(ClassicMetric):
    """This class checks for an exact match."""

    def __evaluate__(self, answer_text, target_text, is_valid):
        if not is_valid:
            return "none"
        if target_text == answer_text:
            return "correct"
        else:
            return "incorrect"


class CaseInsensitiveMatch(ExactMatch):
    """This class checks for a case-insensitive, but otherwise exact match."""

    def __evaluate__(self, answer_text, target_text, is_valid):
        return super().__evaluate__(str(answer_text).lower(), str(target_text).lower(), is_valid)


class IdentityMetric(Metric):

    def evaluate(self, df):
        return df
