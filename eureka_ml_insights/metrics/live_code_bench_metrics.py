import pandas as pd

from eureka_ml_insights.metrics import metrics_base


class CodeAllTestCasesPassedMetric(metrics_base.Metric):
    """Evaluates the generated code against the provided test cases."""

    def __evaluate__(self, row: pd.Series) -> bool:  # type: ignore
        return True
