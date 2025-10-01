""" Main script to run the pipeline for the specified experiment config class name. """

import argparse
import logging
import os
import sys

from eureka_ml_insights import user_configs as configs
from eureka_ml_insights.configs.model_configs import (
    OAI_GPT4O_2024_11_20_CONFIG,
)
from eureka_ml_insights.configs import model_configs as model_configs
from eureka_ml_insights.core import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def import_from_path(module_path, class_name):
    """
    Dynamically import a class from a module path.
    """
    sys.path.append(os.path.dirname(os.path.abspath(module_path)))
    print(sys.path)
    import importlib.util

    spec = importlib.util.spec_from_file_location("experiment_config", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Get the experiment config class from the module
    if hasattr(module, class_name):
        return getattr(module, class_name)
        logging.info(f"Using experiment config class {class_name} from {module_path}.")
    else:
        raise ValueError(f"Experiment config class {class_name} not found in {module_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pipeline for the specified experiment config class name.")
    parser.add_argument("--exp_config", type=str, help="The name of the experiment config class to run.", required=True)
    parser.add_argument("--exp_config_path", type=str, help="Path to the experiment config file.", default=None)
    parser.add_argument(
        "--model_config", type=str, nargs="?", help="The name of the model config to use.", default=None
    )
    parser.add_argument(
        "--eval_model_config", type=str, nargs="?", help="The name of the model config to use.", default=None
    )
    parser.add_argument("--model_name", type=str, help="The name of the deployed vllm model to use.", default=None)
    parser.add_argument(
        "--exp_logdir", type=str, help="The name of the subdirectory in which to save the logs.", default=None
    )
    parser.add_argument(
        "--resume_from", type=str, help="The path to the inference_result.jsonl to resume from.", default=None
    )
    parser.add_argument("--offline_model", action="store_true", help="Use an offline model for inference.")
    parser.add_argument("--offline_file_path", type=str, help="The path to the offline file to use.", default=None)
    parser.add_argument("--local_vllm", action="store_true", help="Deploy/use local vllm for inference.")
    parser.add_argument("--ports", type=str, nargs="*", help="Ports where vllm model is already hosted.", default=None)
    parser.add_argument("--num_servers", type=int, help="Number of servers to deploy.", default=None)
    parser.add_argument("--only_data_processing", action="store_true", help="Only run data processing.", default=None)
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

    if args.local_vllm:
        from eureka_ml_insights.configs.config import ModelConfig
        from eureka_ml_insights.models import LocalVLLMModel

        if args.model_config:
            try:
                model_config = getattr(model_configs, args.model_config)
            except AttributeError:
                raise ValueError(f"Model config class {args.model_config} not found.")
            for arg in ["ports", "num_servers", "model_name"]:
                # If command line args are provided, override the corresponding model_config init_args key.
                if getattr(args, arg) is not None:
                    model_config.init_args[arg] = getattr(args, arg)
            init_args["model_config"] = model_config
            # Logic above is that certain deployment parameters like ports and num_servers
            # can be variable and so we allow them to be overridden by command line args.
        else:
            # If there's no model config provided, create one. Model name is required in this case.
            if args.model_name is None:
                raise ValueError(
                    "Commandline argument --model_name is required when using --local_vllm and no --model_config is provided."
                )

            init_args["model_config"] = ModelConfig(
                LocalVLLMModel, {"model_name": args.model_name, "ports": args.ports, "num_servers": args.num_servers}
            )

    if args.offline_model:
        from eureka_ml_insights.configs.config import ModelConfig
        from eureka_ml_insights.models import OfflineFileModel

        if args.model_name is None or args.offline_file_path is None:
            raise ValueError(
                "Commandline argument --model_name and --offline_file_path are required when using --offline_model."
            )

        init_args["model_config"] = ModelConfig(
            OfflineFileModel, {"model_name": args.model_name, "file_path": args.offline_file_path}
        )

    elif args.model_config:
        try:
            init_args["model_config"] = getattr(model_configs, args.model_config)
        except AttributeError:
            raise ValueError(f"Model config class {args.model_config} not found.")
    if args.eval_model_config:
        try:
            init_args["eval_model_config"] = getattr(model_configs, args.eval_model_config)
        except AttributeError:
            raise ValueError(f"Model config class {args.eval_model_config} not found.")
    else:
        logging.warning("No eval_model_config provided. Using OAI_GPT4O_2024_11_20_CONFIG for eval related LLM calls if needed.")
        init_args["eval_model_config"] = OAI_GPT4O_2024_11_20_CONFIG

    if args.resume_from:
        init_args["resume_from"] = args.resume_from

    if experiment_config_class in dir(configs):
        experiment_config_class = getattr(configs, experiment_config_class)
    else:
        # If the experiment_config_class is not found in the configs module, try to import it from args.exp_config_path.
        if args.exp_config_path:
            experiment_config_class = import_from_path(args.exp_config_path, args.exp_config)
        else:
            raise ValueError(f"Experiment config class {args.exp_config} not found.")
    pipeline_config = experiment_config_class(exp_logdir=args.exp_logdir, **init_args).pipeline_config
    logging.info(f"Saving experiment logs in {pipeline_config.log_dir}.")

    if args.only_data_processing:
        from eureka_ml_insights.configs import PipelineConfig

        pipeline_config = PipelineConfig([pipeline_config.component_configs[0]], pipeline_config.log_dir)
        logging.info("Running only first (data processing) component of the pipeline. Please verify with pipeline")

    pipeline = Pipeline(pipeline_config.component_configs, pipeline_config.log_dir)
    pipeline.run()

    if args.local_vllm:
        from eureka_ml_insights.models.models import (
            _LocalVLLMDeploymentHandler,
        )

        _LocalVLLMDeploymentHandler.shutdown_servers()
