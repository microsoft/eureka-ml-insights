""" Main script to run the pipeline for the specified experiment config class name. """

import argparse
import logging

from eureka_ml_insights import configs
from eureka_ml_insights.configs import model_configs
from eureka_ml_insights.core import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pipeline for the specified experiment config class name.")
    parser.add_argument("--exp_config", type=str, help="The name of the experiment config class to run.", required=True)
    parser.add_argument(
        "--model_config", type=str, nargs="?", help="The name of the model config to use.", default=None
    )
    parser.add_argument(
        "--exp_logdir", type=str, help="The name of the subdirectory in which to save the logs.", default=None
    )
    parser.add_argument(
        "--resume_from", type=str, help="The path to the inference_result.jsonl to resume from.", default=None
    )
    args = parser.parse_args()

    experiment_config_class = args.exp_config
    init_args = {}
    if args.model_config:
        try:
            init_args["model_config"] = getattr(model_configs, args.model_config)
        except AttributeError:
            raise ValueError(f"Model config class {args.model_config} not found.")

    if args.resume_from:
        init_args["resume_from"] = args.resume_from

    if experiment_config_class in dir(configs):
        experiment_config_class = getattr(configs, experiment_config_class)
    else:
        raise ValueError(f"Experiment config class {experiment_config_class} not found.")
    pipeline_config = experiment_config_class(exp_logdir=args.exp_logdir, **init_args).pipeline_config
    logging.info(f"Saving experiment logs in {pipeline_config.log_dir}.")
    pipeline = Pipeline(pipeline_config.component_configs, pipeline_config.log_dir)
    pipeline.run()
