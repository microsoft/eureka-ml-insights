import argparse
from azure.ai.ml import MLClient
from azure.ai.ml import command, Output, Input
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential


def run_job(
        experiment_config_name,
        model_config_name,
        exp_logdir,
        environment_name,
        environment_version,
        subscription_id,
        resource_group_name,
        workspace_name,
        resume_from=None):
    
    pre_command = "import nltk; nltk.download('punkt')"
    job = None
    if resume_from is not None:
        resume_from_string = '${{inputs.resume_from}}/inference_result/inference_result.jsonl'
        python_commandline_string = f'python -c "{pre_command}" && python ms_main.py --exp_config {experiment_config_name} --model_config {model_config_name} --exp_logdir {exp_logdir} --resume_from {resume_from_string}'
        shell_commandline_string = 'cp -R logs/ ${{outputs.eureka_logs}}'
        job = command(
            inputs={
                "resume_from": Input(path="azureml:FlenQA_Resume:1", type="uri_folder")
            },
            code="./",
            command=f"{python_commandline_string} && {shell_commandline_string}",
            environment=f"{environment_name}:{environment_version}",
            compute="cpucluster",
            display_name="eureka-ml",
            experiment_name=f"eureka-ml-{experiment_config_name}-experiment",
            outputs={"eureka_logs": Output(type="uri_folder")}
            )
    else:
        python_commandline_string = f'python -c "{pre_command}" && python ms_main.py --exp_config {experiment_config_name} --model_config {model_config_name} --exp_logdir {exp_logdir}'
        shell_commandline_string = 'cp -R logs/ ${{outputs.eureka_logs}}'
        job = command(
            code="./",
            command=f"{python_commandline_string} && {shell_commandline_string}",
            environment=f"{environment_name}:{environment_version}",
            compute="cpucluster",
            display_name="eureka-ml",
            experiment_name=f"eureka-ml-{experiment_config_name}-experiment",
            outputs={"eureka_logs": Output(type="uri_folder")}
            )

    ws_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
        logging_enable=False,
    )

    returned_job = ws_client.create_or_update(job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kick off an Eureka evaluation job within AML")
    parser.add_argument("--experiment_config_name", type=str, help="Eureka Experiment Pipeline Config Name", required=True)
    parser.add_argument("--model_config_name", type=str, help="Eureka Model Config Name", required=True)
    parser.add_argument("--exp_logdir", type=str, help="Experiment log dir", required=True)
    parser.add_argument(
        "--environment_name", type=str, help="The name of the environment", default="eurekaenvironment", required=False
    )
    parser.add_argument(
        "--environment_version", type=int, help="The environment version", default=1, required=False
    )
    parser.add_argument("--subscription_id", type=str, help="Azure Subscription ID", required=True)
    parser.add_argument("--resource_group_name", type=str, help="Azure Resource Group Name", required=True)
    parser.add_argument("--workspace_name", type=str, help="Azure Workspace Name", required=True)
    parser.add_argument("--resume_from", type=str, help="Resume from", default=None,  required=False)
    args = parser.parse_args()

    run_job(
        args.experiment_config_name,
        args.model_config_name,
        args.exp_logdir,
        args.environment_name,
        args.environment_version,
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
        resume_from = args.resume_from
    )