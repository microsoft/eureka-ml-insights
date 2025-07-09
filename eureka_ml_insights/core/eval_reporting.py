# report component has data reader, list of metrics, list of visualizers, and list of writers
"""This module provides functionality to read data, compute metrics, visualize results, and write reports
through the EvalReporting class and its supporting components."""

import json
import os

from eureka_ml_insights.data_utils import NumpyEncoder
from eureka_ml_insights.metrics import Reporter

from .pipeline import Component


class EvalReporting(Component):
    """Reads data from a given dataset, computes metrics, writes results, and passes results to a reporter.

    Attributes:
        data_reader: An instance of a data reader that loads the dataset.
        metric: An instance of a metric calculator to evaluate the data, if specified.
        reporter: A Reporter object responsible for generating the final report.
    """

    def __init__(
        self, data_reader_config, output_dir, metric_config=None, aggregator_configs=None, visualizer_configs=None
    ):
        """Initializes an EvalReporting instance.

        Args:
            data_reader_config (object): Configuration for the data reader, including class and initialization arguments.
            output_dir (str): Directory where output files will be written.
            metric_config (object, optional): Configuration for the metric calculator.
            aggregator_configs (list, optional): A list of aggregator configurations.
            visualizer_configs (list, optional): A list of visualizer configurations.
        """
        super().__init__(output_dir)
        self.data_reader = data_reader_config.class_name(**data_reader_config.init_args)
        self.metric = None
        if metric_config is not None:
            self.metric = metric_config.class_name(**metric_config.init_args)
        self.reporter = Reporter(output_dir, aggregator_configs, visualizer_configs)

    @classmethod
    def from_config(cls, config):
        """Creates an EvalReporting instance from a configuration object.

        Args:
            config (object): An object containing all necessary configuration for EvalReporting.

        Returns:
            EvalReporting: An initialized instance of EvalReporting.
        """
        return cls(
            data_reader_config=config.data_reader_config,
            output_dir=config.output_dir,
            metric_config=config.metric_config,
            aggregator_configs=config.aggregator_configs,
            visualizer_configs=config.visualizer_configs,
        )

    def run(self):
        """Executes the reporting pipeline.

        Loads the dataset, evaluates it with the metric (if provided), writes the metric
        results to a file (if metric_config is specified), and then generates a report.
        """
        df = self.data_reader.load_dataset()
        metric_result = df
        if self.metric:
            metric_result = self.metric.evaluate(df)
            # write results in the output directory in a file names metric_results.jsonl
            metric_results_file = os.path.join(self.output_dir, "metric_results.jsonl")
            with open(metric_results_file, "w", encoding="utf-8") as writer:
                for _, row in metric_result.iterrows():
                    content = row.to_dict()
                    writer.write(
                        json.dumps(content, ensure_ascii=False, separators=(",", ":"), cls=NumpyEncoder) + "\n"
                    )
        # generate reports
        self.reporter.generate_report(metric_result)
