# report component has data reader, list of metrics, list of visualizers, and list of writers
import json
import os

from eureka_ml_insights.data_utils import NumpyEncoder
from eureka_ml_insights.metrics import Reporter

from .pipeline import Component


class EvalReporting(Component):
    """This class reads the data from a given dataset, calculates the metrics and writes its results,
    and passes the results to reporter."""

    def __init__(
        self, data_reader_config, output_dir, metric_config=None, aggregator_configs=None, visualizer_configs=None
    ):
        super().__init__(output_dir)
        self.data_reader = data_reader_config.class_name(**data_reader_config.init_args)
        self.metric = None
        if metric_config is not None:
            self.metric = metric_config.class_name(**metric_config.init_args)
        self.reporter = Reporter(output_dir, aggregator_configs, visualizer_configs)

    @classmethod
    def from_config(cls, config):
        return cls(
            data_reader_config=config.data_reader_config,
            output_dir=config.output_dir,
            metric_config=config.metric_config,
            aggregator_configs=config.aggregator_configs,
            visualizer_configs=config.visualizer_configs,
        )

    def run(self):
        df = self.data_reader.load_dataset()
        metric_result = df
        if self.metric:
            metric_result = self.metric.evaluate(df)
            # write results in the output directory in a file names metric_resutls.jsonl
            metric_results_file = os.path.join(self.output_dir, "metric_results.jsonl")
            with open(metric_results_file, "w", encoding="utf-8") as writer:
                for _, row in metric_result.iterrows():
                    content = row.to_dict()
                    writer.write(
                        json.dumps(content, ensure_ascii=False, separators=(",", ":"), cls=NumpyEncoder) + "\n"
                    )
        # generate reports
        self.reporter.generate_report(metric_result)
