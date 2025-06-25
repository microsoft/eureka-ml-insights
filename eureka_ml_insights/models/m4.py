<module_content>
"""This module provides classes and functions to handle and interact with language models using vLLM, Claude, and test models."""

@dataclass
class VLLMModel(Model):
    """Runs a self-hosted language model via vLLM APIs.

    Uses the chat() functionality of vLLM, which applies a template included
    in the HF model files. If the model files do not include a template, no
    template will be applied.

    Attributes:
        model_name (str): Name or path of the model.
        trust_remote_code (bool): Whether to trust custom code from the remote model.
        tensor_parallel_size (int): Number of tensor parallel instances.
        dtype (str): Data type used by the model.
        quantization (str): Quantization method used by the model.
        seed (int): Random seed.
        gpu_memory_utilization (float): Fraction of GPU memory to use.
        cpu_offload_gb (float): Amount of CPU offloading in GB.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability threshold.
        top_k (int): Top-k sampling cutoff.
        max_tokens (int): Maximum number of tokens in the generated response.
    """

    model_name: str = None
    trust_remote_code: bool = False
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    quantization: str = None
    seed: int = 0
    gpu_memory_utilization: float = 0.9
    cpu_offload_gb: float = 0

    temperature: float = 0.001
    top_p: float = 0.95
    top_k: int = -1
    max_tokens: int = 2000

    def __post_init__(self):
        """Initializes the vLLM model after dataclass fields are set.

        Returns:
            None
        """
        # vLLM automatically picks an available devices when get_model() is called
        self.get_model()

    def get_model(self):
        """Initializes and stores the LLM instance from vLLM.

        Returns:
            None
        """
        from vllm import LLM

        self.model = LLM(
            model=self.model_name,
            trust_remote_code=self.trust_remote_code,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
            quantization=self.quantization,
            seed=self.seed,
            gpu_memory_utilization=self.gpu_memory_utilization,
            cpu_offload_gb=self.cpu_offload_gb,
        )

    def _generate(self, text_prompt, query_images=None):
        """Generates a response from the model using the vLLM chat interface.

        Args:
            text_prompt (str): The prompt for the model.
            query_images (optional): Additional images to pass to the model. Defaults to None.

        Returns:
            dict: A dictionary containing "model_output" and "response_time".
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
        )

        start_time = time.time()
        outputs = self.model.chat(text_prompt, sampling_params)
        end_time = time.time()

        model_output = outputs[0].outputs[0].text
        response_time = end_time - start_time
        return {
            "model_output": model_output,
            "response_time": response_time,
        }

    def generate(self, text_prompt, query_images=None, system_message=None):
        """Generates a response from the model.

        Args:
            text_prompt (str): The prompt for the model.
            query_images (optional): Additional images to pass to the model. Defaults to None.
            system_message (optional): A system message to prepend. Defaults to None.

        Returns:
            dict: A dictionary with the generated response, including "model_output",
                "is_valid", "response_time", and "n_output_tokens".
        """
        response_dict = {}
        model_output = None
        response_time = None
        is_valid = False
        if text_prompt:
            messages = self.create_request(text_prompt, system_message)
            try:
                model_response = self._generate(messages, query_images=query_images)

                if model_response:
                    response_dict.update(model_response)
                    model_output = model_response["model_output"]
                    response_time = model_response["response_time"]
                is_valid = True

            except Exception as e:
                logging.warning(e)
                is_valid = False

        response_dict.update(
            {
                "model_output": model_output,
                "is_valid": is_valid,
                "response_time": response_time,
                "n_output_tokens": self.count_tokens(model_output, is_valid),
            }
        )
        return response_dict

    def create_request(self, text_prompt, system_message=None):
        """Creates a list of messages suitable for the vLLM chat interface.

        Args:
            text_prompt (str): The user's prompt.
            system_message (optional): A system message to prepend. Defaults to None.

        Returns:
            list: A list of dictionaries representing messages.
        """
        if system_message:
            return [
                {"role": "system", "content": system_message},
                {"role": "user", "content": text_prompt},
            ]
        else:
            return [{"role": "user", "content": text_prompt}]


class _LocalVLLMDeploymentHandler:
    """Handles the deployment of vLLM servers.

    Attributes:
        logs (list): References to logs of the servers, which contain PIDs for shutdown.
    """

    # Chose against dataclass here so we have the option to accept kwargs
    # and pass them to the vLLM deployment script.

    # Used to store references to logs of the servers, since those contain PIDs for shutdown.
    logs = []

    def __init__(
        self,
        model_name: str = None,
        num_servers: int = 1,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: str = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        cpu_offload_gb: float = 0,
        ports: list = None,
    ):
        """Initializes the local vLLM deployment handler.

        Args:
            model_name (str): Name or path of the model to deploy.
            num_servers (int): Number of servers to deploy.
            trust_remote_code (bool): Whether to trust remote code.
            tensor_parallel_size (int): Number of tensor parallel instances.
            pipeline_parallel_size (int): Number of pipeline parallel instances.
            dtype (str): Data type used by the model.
            quantization (str): Quantization method used.
            seed (int): Random seed.
            gpu_memory_utilization (float): Fraction of GPU memory to use.
            cpu_offload_gb (float): Amount of CPU offloading in GB.
            ports (list): List of ports at which servers will be hosted.
        Raises:
            ValueError: If model_name is not specified.
            RuntimeError: If not all servers can be started.
        """
        if not model_name:
            raise ValueError("LocalVLLM model_name must be specified.")
        self.model_name = model_name
        self.num_servers = num_servers
        self.trust_remote_code = trust_remote_code
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.dtype = dtype
        self.quantization = quantization
        self.seed = seed
        self.gpu_memory_utilization = gpu_memory_utilization
        self.cpu_offload_gb = cpu_offload_gb

        self.ports = ports
        self.session = requests.Session()
        self.clients = self._get_clients()
        if len(self.clients) != self.num_servers:
            raise RuntimeError(f"Failed to start all servers.")

    def _get_clients(self):
        """Gets clients to access vLLM servers.

        Checks for running servers and deploys them if necessary.

        Returns:
            list: A list of OpenAIClient objects for each healthy server.
        """
        from openai import OpenAI as OpenAIClient

        # If the user passes ports, check if the servers are running and populate clients accordingly.
        if self.ports:
            healthy_server_urls = ["http://0.0.0.0:" + port + "/v1" for port in self.get_healthy_ports()]
            if len(healthy_server_urls) > 0:
                logging.info(f"Found {len(healthy_server_urls)} healthy servers.")
                return [OpenAIClient(base_url=url, api_key="none") for url in healthy_server_urls]

        # Even if the user doesn't pass ports, we can check if there happen to be deployed servers.
        # There is no guarantee that the servers are hosting the correct model.
        self.ports = [str(8000 + i) for i in range(self.num_servers)]
        healthy_server_urls = ["http://0.0.0.0:" + port + "/v1" for port in self.get_healthy_ports()]
        if len(healthy_server_urls) == self.num_servers:
            logging.info(f"Found {len(healthy_server_urls)} healthy servers.")
            return [OpenAIClient(base_url=url, api_key="none") for url in healthy_server_urls]

        # If that didn't work, let's deploy and wait for servers to come online.
        self.deploy_servers()
        server_start_time = time.time()
        while time.time() - server_start_time < 600:
            time.sleep(10)
            healthy_ports = self.get_healthy_ports()
            if len(healthy_ports) == self.num_servers:
                logging.info(f"All {self.num_servers} servers are online.")
                healthy_server_urls = ["http://0.0.0.0:" + port + "/v1" for port in healthy_ports]
                return [OpenAIClient(base_url=url, api_key="none") for url in healthy_server_urls]
            else:
                logging.info(f"Waiting for {self.num_servers - len(healthy_ports)} more servers to come online.")
        # If we get here, we timed out waiting for servers to come online.
        raise RuntimeError(f"Failed to start all servers.")

    def get_healthy_ports(self) -> list[str]:
        """Checks if vLLM servers are running.

        Returns:
            list[str]: A list of ports that are healthy and running.
        """
        healthy_ports = []
        for port in self.ports:
            try:
                self.session.get("http://0.0.0.0:" + port + "/health")
                healthy_ports.append(port)
            except:
                pass
        return healthy_ports

    def deploy_servers(self):
        """Deploys vLLM servers in background threads using the specified parameters.

        Returns:
            None
        """
        logging.info(f"No vLLM servers are running. Starting {self.num_servers} new servers at {self.ports}.")
        import datetime
        import os

        gpus_per_port = self.pipeline_parallel_size * self.tensor_parallel_size
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        log_dir = os.path.join("logs", "local_vllm_deployment_logs", f"{date}")
        os.makedirs(log_dir)

        for index in range(self.num_servers):
            port = 8000 + index
            log_file = os.path.join(log_dir, f"{port}.log")
            self.logs.append(log_file)
            background_thread = threading.Thread(target=lambda: self.deploy_server(index, gpus_per_port, log_file))
            background_thread.daemon = True
            background_thread.start()

    def deploy_server(self, index: int, gpus_per_port: int, log_file: str):
        """Deploys a single vLLM server.

        Uses gpus_per_port GPUs starting at index*gpus_per_port.

        Args:
            index (int): Index of the server to deploy.
            gpus_per_port (int): Number of GPUs to allocate per server.
            log_file (str): File path to store the server logs.

        Returns:
            None
        """
        import subprocess

        port = 8000 + index
        first_gpu = index * gpus_per_port
        last_gpu = first_gpu + gpus_per_port - 1
        devices = ",".join(str(gpu_num) for gpu_num in range(first_gpu, last_gpu + 1))

        command = [
            "CUDA_VISIBLE_DEVICES=" + devices,
            "vllm serve",
            self.model_name,
            "--port",
            str(port),
            "--tensor_parallel_size",
            str(self.tensor_parallel_size),
            "--pipeline_parallel_size",
            str(self.pipeline_parallel_size),
            "--dtype",
            self.dtype,
            "--seed",
            str(self.seed),
            "--gpu_memory_utilization",
            str(self.gpu_memory_utilization),
            "--cpu_offload_gb",
            str(self.cpu_offload_gb),
        ]
        if self.quantization:
            command.append("--quantization")
            command.append(self.quantization)
        if self.trust_remote_code:
            command.append("--trust_remote_code")
        command = " ".join(command)
        logging.info(f"Running command: {command}")
        with open(log_file, "w") as log_writer:
            subprocess.run(command, shell=True, stdout=log_writer, stderr=log_writer)

    @classmethod
    def shutdown_servers(cls):
        """Shuts down all vLLM servers deployed during this run.

        Returns:
            None
        """
        import os
        import re
        import signal

        for log_file in cls.logs:
            with open(log_file, "r") as f:
                for line in f:
                    if "Started server process" in line:
                        match = re.search(r"\d+", line)
                        if match:
                            pid = int(match.group())
                            logging.info(f"Shutting down server with PID {pid}.")
                            os.kill(pid, signal.SIGINT)
                            break


local_vllm_model_lock = threading.Lock()
local_vllm_deployment_handlers: dict[str, _LocalVLLMDeploymentHandler] = {}


@dataclass
class LocalVLLMModel(OpenAICommonRequestResponseMixIn, EndpointModel):
    """Represents a local vLLM server deployment.

    In case the servers are already deployed, specify the model_name and the
    ports at which the servers are hosted. Otherwise, instantiating this class
    will initiate a server deployment using any specified deployment parameters.

    Attributes:
        model_name (str): Name or path of the model.
        num_servers (int): Number of servers to deploy.
        trust_remote_code (bool): Whether to trust remote code.
        tensor_parallel_size (int): Number of tensor parallel instances.
        pipeline_parallel_size (int): Number of pipeline parallel instances.
        dtype (str): Data type used by the model.
        quantization (str): Quantization method used by the model.
        seed (int): Random seed.
        gpu_memory_utilization (float): Fraction of GPU memory to use.
        cpu_offload_gb (float): Amount of CPU offloading in GB.
        ports (list): Ports at which servers are hosted or will be hosted.
        handler (_LocalVLLMDeploymentHandler): Deployment handler instance.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability threshold.
        top_k (int): Top-k sampling cutoff.
        max_tokens (int): Maximum number of tokens in the generated response.
        frequency_penalty (float): Frequency penalty.
        presence_penalty (float): Presence penalty.
    """

    model_name: str = None

    # Deployment parameters
    num_servers: int = 1
    trust_remote_code: bool = False
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    dtype: str = "auto"
    quantization: str = None
    seed: int = 0
    gpu_memory_utilization: float = 0.9
    cpu_offload_gb: float = 0

    # Deployment handler
    ports: list = None
    handler: _LocalVLLMDeploymentHandler = None

    # Inference parameters
    temperature: float = 0.01
    top_p: float = 0.95
    top_k: int = -1
    max_tokens: int = 2000
    frequency_penalty: float = 0
    presence_penalty: float = 0

    def __post_init__(self):
        """Initializes the local vLLM model deployment.

        Raises:
            ValueError: If model_name is not specified.
        """
        if not self.model_name:
            raise ValueError("LocalVLLM model_name must be specified.")
        self.handler = self._get_local_vllm_deployment_handler()

    @property
    def client(self):
        """Randomly selects a client from the list of deployed servers.

        Returns:
            OpenAIClient: A client for sending requests to the vLLM server.
        """
        return random.choice(self.handler.clients)

    def _get_local_vllm_deployment_handler(self):
        """Gets or creates a local vLLM deployment handler for this model.

        Returns:
            _LocalVLLMDeploymentHandler: The handler responsible for server deployment.
        """
        if self.model_name not in local_vllm_deployment_handlers:
            with local_vllm_model_lock:
                if self.model_name not in local_vllm_deployment_handlers:
                    local_vllm_deployment_handlers[self.model_name] = _LocalVLLMDeploymentHandler(
                        model_name=self.model_name,
                        num_servers=self.num_servers,
                        trust_remote_code=self.trust_remote_code,
                        pipeline_parallel_size=self.pipeline_parallel_size,
                        tensor_parallel_size=self.tensor_parallel_size,
                        dtype=self.dtype,
                        quantization=self.quantization,
                        seed=self.seed,
                        gpu_memory_utilization=self.gpu_memory_utilization,
                        cpu_offload_gb=self.cpu_offload_gb,
                        ports=self.ports,
                    )

        return local_vllm_deployment_handlers[self.model_name]

    def handle_request_error(self, e):
        """Handles any request error that occurs during inference.

        Args:
            e (Exception): The exception raised during inference.

        Returns:
            bool: Always returns False indicating the request was not successful.
        """
        return False


@dataclass
class ClaudeModel(EndpointModel, KeyBasedAuthMixIn):
    """Interacts with Claude models through the Python API.

    Attributes:
        model_name (str): Name of the Claude model.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum number of tokens in the generated response.
        top_p (float): Nucleus sampling probability threshold.
        timeout (int): Request timeout in seconds.
    """

    model_name: str = None
    temperature: float = 0
    max_tokens: int = 2000
    top_p: float = 0.95
    timeout: int = 60

    def __post_init__(self):
        """Initializes the Claude client after dataclass fields are set.

        Returns:
            None
        """
        super().__post_init__()
        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def create_request(self, text_prompt, query_images=None, system_message=None, previous_messages=None):
        """Creates a request payload for Claude.

        Args:
            text_prompt (str): The user's prompt.
            query_images (optional): Additional images to encode and send. Defaults to None.
            system_message (optional): A system message to prepend. Defaults to None.
            previous_messages (optional): A list of previous conversation messages. Defaults to None.

        Returns:
            dict: A dictionary containing 'messages' and optionally 'system'.
        """
        messages = []
        user_content = text_prompt
        if previous_messages:
            messages.extend(previous_messages)
        if query_images:
            encoded_images = self.base64encode(query_images)
            user_content = [
                {"type": "text", "text": text_prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": encoded_images[0],
                    },
                },
            ]
        messages.append({"role": "user", "content": user_content})

        if system_message:
            return {"messages": messages, "system": system_message}
        else:
            return {"messages": messages}

    def get_response(self, request):
        """Sends a request to the Claude model and gets a response.

        Args:
            request (dict): The request payload to be sent.

        Returns:
            dict: A dictionary containing "model_output", "response_time", and optionally "usage".
        """
        start_time = time.time()
        completion = self.client.messages.create(
            model=self.model_name,
            **request,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        end_time = time.time()
        model_output = completion.content[0].text
        response_time = end_time - start_time
        response_dict = {
            "model_output": model_output,
            "response_time": response_time,
        }
        if hasattr(completion, "usage"):
            response_dict.update({"usage": completion.usage.to_dict()})
        return response_dict

    def handle_request_error(self, e):
        """Handles any request error that occurs during Claude inference.

        Args:
            e (Exception): The exception raised during inference.

        Returns:
            bool: Always returns False indicating the request was not successful.
        """
        return False


@dataclass
class ClaudeReasoningModel(ClaudeModel):
    """Interacts with Claude reasoning models through the Python API.

    Allows usage of a 'thinking' parameter for advanced reasoning.

    Attributes:
        model_name (str): Name of the Claude reasoning model.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum number of tokens in the generated response.
        timeout (int): Request timeout in seconds.
        thinking_enabled (bool): Whether thinking mode is enabled.
        thinking_budget (int): Token budget for thinking mode.
        top_p (float): This parameter is not supported and will be ignored.
    """

    model_name: str = None
    temperature: float = 1.0
    max_tokens: int = 20000
    timeout: int = 600
    thinking_enabled: bool = True
    thinking_budget: int = 16000
    top_p: float = None

    def get_response(self, request):
        """Sends a request to the Claude reasoning model and gets a response.

        Args:
            request (dict): The request payload to be sent.

        Returns:
            dict: A dictionary containing "model_output", "response_time",
                "thinking_output", "redacted_thinking_output", and optionally "usage".
        """
        model_output = None
        response_time = None
        thinking_output = None
        redacted_thinking_output = None
        response_dict = {}
        if self.top_p is not None:
            logging.warning("top_p is not supported for claude reasoning models as of 03/08/2025. It will be ignored.")

        start_time = time.time()
        thinking = {"type": "enabled", "budget_tokens": self.thinking_budget} if self.thinking_enabled else None
        completion = self.client.messages.create(
            model=self.model_name,
            **request,
            temperature=self.temperature,
            thinking=thinking,
            max_tokens=self.max_tokens,
        )
        end_time = time.time()

        # Loop through completion.content to find the text output
        for content in completion.content:
            if content.type == "text":
                model_output = content.text
            elif content.type == "thinking":
                thinking_output = content.thinking
            elif content.type == "redacted_thinking":
                redacted_thinking_output = content.data

        response_time = end_time - start_time
        response_dict = {
            "model_output": model_output,
            "response_time": response_time,
            "thinking_output": thinking_output,
            "redacted_thinking_output": redacted_thinking_output,
        }
        if hasattr(completion, "usage"):
            response_dict.update({"usage": completion.usage.to_dict()})
        return response_dict


@dataclass
class TestModel(Model):
    """A model used for testing purposes only.

    This model waits for a specified time and returns a predetermined response.
    """

    def generate(self, text_prompt, **kwargs):
        """Generates a test response.

        Args:
            text_prompt (str): The input prompt (unused).

        Returns:
            dict: A dictionary containing "model_output", "is_valid", "response_time",
                and "n_output_tokens".
        """
        output = "This is a test response."
        is_valid = True
        return {
            "model_output": output,
            "is_valid": is_valid,
            "response_time": 0.1,
            "n_output_tokens": self.count_tokens(output, is_valid),
        }

</module_content>