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
        # TODO this would break if the first row does not have all the metrics, e.g. due invalid inference results
        for metric_name in data[composite_metric_col][0]:
            data[self.__class__.__name__ + "_" + metric_name] = data.apply(
                lambda row: (
                    row[composite_metric_col].get(metric_name, None)
                    if isinstance(row[composite_metric_col], dict)
                    else None
                ),
                axis=1,
            )
        data.drop(composite_metric_col, axis=1, inplace=True)
        return data


class ClassicMetric(Metric):
    """This class is a base class for metrics that require ground truths and predictions."""

    def __init__(self, model_output_col: str = "model_output"):
        super().__init__()
        self.model_output_col = model_output_col

    def validate_data(self, data):
        """This method checks if the data has the required fields."""
        super().validate_data(data)
        assert "ground_truth" in data.columns, "Data does not have 'ground_truth' field."
        assert self.model_output_col in data.columns, f"Data does not have '{self.model_output_col}' field."
        return True

    def evaluate(self, data):
        self.validate_data(data)
        data[self.__class__.__name__ + "_result"] = data.apply(
            lambda x: self.__evaluate__(x[self.model_output_col], x["ground_truth"], x["is_valid"]), axis=1
        )
        return data


class DetectionMetric(Metric):
    """
    This class is for the detection metric, where access is needed to the image
    and ground truth is stored in an external file.
    """

    def __init__(self, model_output_col: str = "model_output"):
        super().__init__()
        self.model_output_col = model_output_col

    def validate_data(self, data):
        """This method checks if the data has the required fields."""
        super().validate_data(data)
        assert "id" in data.columns, "Data does not have 'id' field."
        assert self.model_output_col in data.columns, f"Data does not have '{self.model_output_col}' field."
        return True

    def evaluate(self, data):
        self.validate_data(data)

        tqdm.pandas()

        data[self.__class__.__name__ + "_result"] = data.progress_apply(
            lambda x: self.__evaluate__(x["id"], x[self.model_output_col], x["is_valid"]), axis=1
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
            lambda x: self.__evaluate__(
                x[self.model_output_col], x["ground_truth"], x["target_options"], x["is_valid"]
            ),
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
                x[self.model_output_col], x["ground_truth"], x["question_type"], x["target_options"], x["is_valid"]
            ),
            axis=1,
        )
        return data


class SubstringExistsMatch(ClassicMetric):
    """This class checks for a case-insensitive substring match."""

    def __evaluate__(self, answer_text, target_text, is_valid):
        if not is_valid:
            return "none"
        
        # Make sure everything is a strings.
        answer_text = str(answer_text)
        target_text = str(target_text)

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


class MetricBasedVerifier:
    """ This class enables the use of a metric as a verifier, in other words, it wraps a given metric class as a
    data transformation and stores the verification result in a column called "verification_result"."""
    def __init__(self, metric_class, model_output_col: str = "model_output"):
        """
        args:
            metric_class: The class of the metric to be used for verification.
            model_output_col: The name of the column containing the model output.
        """
        self.model_output_col = model_output_col
        self.metric_class = metric_class

    def transform(self, data):
        data = self.metric_class(model_output_col=self.model_output_col).evaluate(data)
        # rename the result column to "verification_reuslt"
        data.rename(columns={self.metric_class.__name__ + "_result": "verification_result"}, inplace=True)
        return data
    

class BboxMetric(Metric):
    """This class is a base class for metrics that require ground truths and predictions."""

    def __init__(self, model_output_col: str = "model_output", normalized: bool = False, xywh: bool = True):
        super().__init__()
        self.model_output_col = model_output_col
        self.normalized = normalized
        self.xywh = xywh

    def validate_data(self, data):
        """This method checks if the data has the required fields."""
        super().validate_data(data)
        assert "bbox" in data.columns, "Data does not have 'bbox' field."
        assert self.model_output_col in data.columns, f"Data does not have '{self.model_output_col}' field."
        return True

    def __evaluate__(self, bbox_answer, bbox, is_valid):
        import ast

        if not is_valid or not bbox_answer:
            return "none"
        
        if self.xywh:
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        else:
            bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]

        click_point = [(bbox_answer[0] + bbox_answer[2]) / 2, (bbox_answer[1] + bbox_answer[3]) / 2]

        # Check if the predicted point falls in the ground truth box
        if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
            return "correct"
        else:
            return "incorrect"

    def evaluate(self, data):
        self.validate_data(data)
        data[self.__class__.__name__ + "_result"] = data.apply(
            lambda x: self.__evaluate__(x[self.model_output_col], x["bbox_normalized"] if self.normalized else x["bbox"], x["is_valid"]), axis=1
        )
        return data    