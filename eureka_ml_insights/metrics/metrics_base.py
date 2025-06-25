from tqdm.auto import tqdm


class Metric:
    """Base class for metrics.

    This class defines the core structure for validating data and evaluating
    a metric on a given dataset. Subclasses should implement the __evaluate__
    method.
    """

    def validate_data(self, data):
        """Validates input data.

        Args:
            data (pd.DataFrame): The input DataFrame to validate.

        Returns:
            bool: True if the data is valid.

        Raises:
            AssertionError: If the 'is_valid' column is missing.
        """
        assert "is_valid" in data.columns, "Data does not have 'is_valid' field."
        return True

    def __evaluate__(self, **kwargs):
        """Evaluates the metric for a single row.

        Kwargs:
            **kwargs: Arbitrary keyword arguments for evaluation.

        Returns:
            Any: The result of the evaluation.

        Raises:
            NotImplementedError: This method must be overridden by subclasses.
        """
        raise NotImplementedError

    def evaluate(self, data):
        """Evaluates the metric on the entire dataset.

        Args:
            data (pd.DataFrame): The input DataFrame on which to run the metric.

        Returns:
            pd.DataFrame: A copy of the DataFrame with metric results appended
            as a new column.
        """
        self.validate_data(data)
        data[self.__class__.__name__ + "_result"] = data.apply(lambda x: self.__evaluate__(x), axis=1)
        return data


class CompositeMetric(Metric):
    """Base class for composite metrics that consist of several submetrics.

    This class returns a dictionary mapping "metric_name" to a value for each
    row of data.
    """

    def evaluate(self, data):
        """Evaluates the composite metric on the given data.

        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with decomposition applied.
        """
        super().evaluate(data)
        data = self.decompose_metric(data)
        return data

    def decompose_metric(self, data):
        """Decomposes the composite metric results into separate columns.

        Args:
            data (pd.DataFrame): The input DataFrame with composite metric results.

        Returns:
            pd.DataFrame: The DataFrame with each submetric's result in a separate column.
        """
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
    """Base class for metrics that require ground truths and predictions."""

    def __init__(self, model_output_col: str = "model_output"):
        """Initializes the ClassicMetric.

        Args:
            model_output_col (str, optional): The name of the column containing
                model outputs. Defaults to "model_output".
        """
        super().__init__()
        self.model_output_col = model_output_col

    def validate_data(self, data):
        """Validates that the required fields exist in the DataFrame.

        Args:
            data (pd.DataFrame): The input DataFrame to validate.

        Returns:
            bool: True if the data is valid.

        Raises:
            AssertionError: If required columns are missing.
        """
        super().validate_data(data)
        assert "ground_truth" in data.columns, "Data does not have 'ground_truth' field."
        assert self.model_output_col in data.columns, f"Data does not have '{self.model_output_col}' field."
        return True

    def evaluate(self, data):
        """Evaluates the metric on the given DataFrame.

        Args:
            data (pd.DataFrame): The input DataFrame on which to run the metric.

        Returns:
            pd.DataFrame: The DataFrame with metric results appended.
        """
        self.validate_data(data)
        data[self.__class__.__name__ + "_result"] = data.apply(
            lambda x: self.__evaluate__(x[self.model_output_col], x["ground_truth"], x["is_valid"]), axis=1
        )
        return data


class DetectionMetric(Metric):
    """Class for detection metrics requiring image access and external ground truth.

    This metric requires additional information such as an image ID and
    references an external ground truth file.
    """

    def __init__(self, model_output_col: str = "model_output"):
        """Initializes the DetectionMetric.

        Args:
            model_output_col (str, optional): The name of the column containing
                model outputs. Defaults to "model_output".
        """
        super().__init__()
        self.model_output_col = model_output_col

    def validate_data(self, data):
        """Validates that the required fields exist in the DataFrame.

        Args:
            data (pd.DataFrame): The input DataFrame to validate.

        Returns:
            bool: True if the data is valid.

        Raises:
            AssertionError: If required columns are missing.
        """
        super().validate_data(data)
        assert "id" in data.columns, "Data does not have 'id' field."
        assert self.model_output_col in data.columns, f"Data does not have '{self.model_output_col}' field."
        return True

    def evaluate(self, data):
        """Evaluates the detection metric on the given DataFrame.

        Args:
            data (pd.DataFrame): The input DataFrame on which to run the metric.

        Returns:
            pd.DataFrame: The DataFrame with detection metric results appended.
        """
        self.validate_data(data)

        tqdm.pandas()

        data[self.__class__.__name__ + "_result"] = data.progress_apply(
            lambda x: self.__evaluate__(x["id"], x[self.model_output_col], x["is_valid"]), axis=1
        )
        return data


class MultipleChoiceMetric(ClassicMetric):
    """Base class for metrics that have a multiple choice answer."""

    def validate_data(self, data):
        """Validates that the required fields exist in the DataFrame.

        Args:
            data (pd.DataFrame): The input DataFrame to validate.

        Returns:
            bool: True if the data is valid.

        Raises:
            AssertionError: If required columns are missing.
        """
        super().validate_data(data)
        assert "target_options" in data.columns, "Data does not have 'target_options' field."
        return True

    def evaluate(self, data):
        """Evaluates the multiple choice metric on the given DataFrame.

        Args:
            data (pd.DataFrame): The input DataFrame on which to run the metric.

        Returns:
            pd.DataFrame: The DataFrame with metric results appended.
        """
        self.validate_data(data)
        data[self.__class__.__name__ + "_result"] = data.apply(
            lambda x: self.__evaluate__(
                x[self.model_output_col], x["ground_truth"], x["target_options"], x["is_valid"]
            ),
            axis=1,
        )
        return data


class MixedQuestionTypeMetric(MultipleChoiceMetric):
    """Base class for metrics that can handle multiple choice or open-ended answers."""

    def validate_data(self, data):
        """Validates that the required fields exist in the DataFrame.

        Args:
            data (pd.DataFrame): The input DataFrame to validate.

        Returns:
            bool: True if the data is valid.

        Raises:
            AssertionError: If required columns are missing.
        """
        super().validate_data(data)
        assert "question_type" in data.columns, "Data does not have 'question_type' field."
        return True

    def evaluate(self, data):
        """Evaluates the metric for both multiple choice and open-ended answers.

        Args:
            data (pd.DataFrame): The input DataFrame on which to run the metric.

        Returns:
            pd.DataFrame: The DataFrame with metric results appended.
        """
        self.validate_data(data)
        data[self.__class__.__name__ + "_result"] = data.apply(
            lambda x: self.__evaluate__(
                x[self.model_output_col], x["ground_truth"], x["question_type"], x["target_options"], x["is_valid"]
            ),
            axis=1,
        )
        return data


class SubstringExistsMatch(ClassicMetric):
    """Checks for a case-insensitive substring match."""

    def __evaluate__(self, answer_text, target_text, is_valid):
        """Evaluates the SubstringExistsMatch metric for a single row.

        Args:
            answer_text (str): The predicted answer text.
            target_text (str): The ground truth text.
            is_valid (bool): Whether the entry is valid.

        Returns:
            str: "correct" if the target text is a substring of the answer text
            (ignoring case), otherwise "incorrect". If invalid, returns "none".
        """
        if not is_valid:
            return "none"
        
        # Make sure everything is strings.
        answer_text = str(answer_text)
        target_text = str(target_text)

        return "correct" if target_text.lower() in answer_text.lower() else "incorrect"


class ExactMatch(ClassicMetric):
    """Checks for an exact match between predicted and ground truth text."""

    def __evaluate__(self, answer_text, target_text, is_valid):
        """Evaluates the ExactMatch metric for a single row.

        Args:
            answer_text (str): The predicted answer text.
            target_text (str): The ground truth text.
            is_valid (bool): Whether the entry is valid.

        Returns:
            str: "correct" if the predicted text exactly matches the ground truth
            text, otherwise "incorrect". If invalid, returns "none".
        """
        if not is_valid:
            return "none"
        if target_text == answer_text:
            return "correct"
        else:
            return "incorrect"


class CaseInsensitiveMatch(ExactMatch):
    """Checks for a case-insensitive exact match between predicted and ground truth text."""

    def __evaluate__(self, answer_text, target_text, is_valid):
        """Evaluates the CaseInsensitiveMatch metric for a single row by converting
        both texts to lowercase.

        Args:
            answer_text (str): The predicted answer text.
            target_text (str): The ground truth text.
            is_valid (bool): Whether the entry is valid.

        Returns:
            str: "correct" if the predicted text matches the ground truth text
            (ignoring case), otherwise "incorrect". If invalid, returns "none".
        """
        return super().__evaluate__(str(answer_text).lower(), str(target_text).lower(), is_valid)


class IdentityMetric(Metric):
    """Returns the data unmodified."""

    def evaluate(self, df):
        """Evaluates the DataFrame by returning it unchanged.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The same DataFrame without modification.
        """
        return df


class MetricBasedVerifier:
    """Wraps a given metric class as a data transformation for verification.

    This class stores the verification result in a column called "verification_result".
    """
    def __init__(self, metric_class, model_output_col: str = "model_output"):
        """Initializes MetricBasedVerifier.

        Args:
            metric_class (Type[Metric]): The class of the metric to be used for verification.
            model_output_col (str, optional): The name of the column containing
                the model output. Defaults to "model_output".
        """
        self.model_output_col = model_output_col
        self.metric_class = metric_class

    def transform(self, data):
        """Applies the wrapped metric to transform the data and stores the result.

        Args:
            data (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The DataFrame with an added "verification_result" column.
        """
        data = self.metric_class(model_output_col=self.model_output_col).evaluate(data)
        # rename the result column to "verification_result"
        data.rename(columns={self.metric_class.__name__ + "_result": "verification_result"}, inplace=True)
        return data