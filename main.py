""" Main script to run the pipeline for the specified experiment config class name. """

import argparse
import logging

from eureka_ml_insights import user_configs as configs
from eureka_ml_insights.configs import pvt_model_configs as model_configs
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
    parser.add_argument("--local_vllm", action="store_true", help="Deploy/use local vllm for inference.")
    parser.add_argument("--ports", type=str, nargs="*", help="Ports where vllm model is already hosted.", default=None)
    parser.add_argument("--num_servers", type=int, help="Number of servers to deploy.", default=None)
    init_args = {}

    # catch any unknown arguments
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        # if every other unknown arg starts with "--", parse the unknown args as key-value pairs in a dict
        if all(arg.startswith("--") for arg in unknown_args[::2]):
            init_args.update(
                {arg[len("--") :]: unknown_args[i + 1] for i, arg in enumerate(unknown_args) if i % 2 == 0}
            )
            logging.info(f"Unknown arguments: {init_args} will be sent to the experiment config class.")
        # else, parse the unknown args as is ie. as a list
        else:
            init_args["unknown_args"] = unknown_args
            logging.info(f"Unknown arguments: {unknown_args} will be sent as is to the experiment config class.")

    experiment_config_class = args.exp_config

    if args.local_vllm and args.model_config:
        from eureka_ml_insights.configs.config import ModelConfig
        from eureka_ml_insights.models import LocalVLLMModel
        try:
            model_config = getattr(model_configs, args.model_config)
            if isinstance(model_config, ModelConfig):
                model_config.init_args["ports"] = args.ports
                model_config.init_args["num_servers"] = args.num_servers if args.num_servers else 1
                init_args["model_config"] = model_config
            # Logic above is that certain deployment parameters like ports and num_servers
            # can be variable and so we allow them to be overridden by command line args.
        except AttributeError:
            # If there's no config, create one.
            init_args["model_config"] = ModelConfig(
                LocalVLLMModel,
                {
                    "model_name": args.model_config,
                    "ports": args.ports,
                    "num_servers": args.num_servers if args.num_servers else 1
                }
            )

    elif args.model_config:
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

    if args.local_vllm:
        from eureka_ml_insights.models.models import _LocalVLLMDeploymentHandler
        _LocalVLLMDeploymentHandler.shutdown_servers()
