"""
Interact with various models, including API-based and HuggingFace models.

This module provides multiple classes that define and manage model interactions, including
chat-based and endpoint-based workflows. It handles text-based requests, token counting,
and authentication for a variety of model endpoints.
"""

import json
import logging
import random
import threading
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass

import anthropic
import requests
import tiktoken
from azure.identity import DefaultAzureCredential, get_bearer_token_provider, AzureCliCredential

from eureka_ml_insights.secret_management import get_secret


@dataclass
class Model(ABC):
    """Base class for models.

    Any model class should inherit from this class and implement the 'generate' method, which returns
    a dictionary containing 'model_output', 'is_valid', and any other relevant information.

    Attributes:
        chat_mode (bool): Whether the model operates in chat mode.
        system_message (str): A system message that can be used by the model.
    """

    chat_mode: bool = False
    system_message: str = None

    @abstractmethod
    def generate(self, text_prompt, *args, **kwargs):
        """Generate a response from the model.

        Args:
            text_prompt (str): The text prompt to generate a response for.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing 'model_output', 'is_valid', and any other fields.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    def count_tokens(self, model_output: str = None, is_valid: bool = False):
        """Count the number of tokens in the response.

        Uses the tiktoken tokenizer to count tokens in the model's response. For more information,
        see:
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

        Args:
            model_output (str, optional): The text response from the model. Defaults to None.
            is_valid (bool, optional): Whether the response is valid. Defaults to False.

        Returns:
            int or None: The number of tokens, or None if the response is invalid or None.
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        if model_output is None or not is_valid:
            return None
        else:
            n_output_tokens = len(encoding.encode(model_output))
            return n_output_tokens

    def base64encode(self, query_images):
        """Encode a list of images as base64 strings.

        Args:
            query_images (list): A list of PIL Images to encode as base64 strings.

        Returns:
            list: A list of base64-encoded string representations of the images.
        """
        import base64
        from io import BytesIO

        encoded_images = []

        for query_image in query_images:
            buffered = BytesIO()
            query_image.save(buffered, format="JPEG")
            base64_bytes = base64.b64encode(buffered.getvalue())
            base64_string = base64_bytes.decode("utf-8")
            encoded_images.append(base64_string)

        return encoded_images


@dataclass
class KeyBasedAuthMixIn:
    """Mixin class for key-based authentication.

    This class handles key-based authentication for models.

    Attributes:
        api_key (str): The API key used for authentication.
        secret_key_params (dict): The parameters used to retrieve the API key from a secret manager, if necessary.
    """

    api_key: str = None
    secret_key_params: dict = None

    def __post_init__(self):
        """Initialize the mixin.

        Ensures that either an API key or secret key parameters are provided.

        Raises:
            ValueError: If neither an API key nor secret key parameters are provided.
        """
        if self.api_key is None and self.secret_key_params is None:
            raise ValueError("Either api_key or secret_key_params must be provided.")
        self.api_key = self.get_api_key()

    def get_api_key(self):
        """Retrieve the API key for key-based authentication.

        If 'api_key' is not directly provided, 'secret_key_params' is used to retrieve
        the API key using 'get_secret'.

        Returns:
            str: The API key.
        """
        if self.api_key is None:
            self.api_key = get_secret(**self.secret_key_params)
        return self.api_key


@dataclass
class EndpointModel(Model):
    """Abstract class for interacting with API-based models.

    Inherits from:
        Model (ABC)

    Attributes:
        chat_mode (bool): Whether the model operates in chat mode (from Model).
        system_message (str): A system message that can be used by the model (from Model).
        num_retries (int): The number of times to retry the request if it fails.
    """

    num_retries: int = 3

    @abstractmethod
    def create_request(self, text_prompt, *args, **kwargs):
        """Create a request for the model.

        Args:
            text_prompt (str): The text prompt to generate the request.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The request object to be sent to the model.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def get_response(self, request):
        """Get the response from the model.

        Args:
            request (Any): The request object created by create_request.

        Returns:
            dict: A dictionary containing the model output and the response time.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def update_chat_history(self, query_text, model_output, *args, **kwargs):
        """Update the chat history with the model response.

        Args:
            query_text (str): The text prompt used to generate the model response.
            model_output (str): The text response from the model.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments (may contain 'previous_messages').

        Returns:
            list: A list of messages representing the updated chat history.
        """
        previous_messages = kwargs.get("previous_messages", [])
        previous_messages.append({"role": "user", "content": query_text})
        previous_messages.append({"role": "assistant", "content": model_output})
        return previous_messages

    def generate(self, query_text, *args, **kwargs):
        """Call the endpoint to generate the model response.

        Args:
            query_text (str): The text prompt to generate the response for.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments (may include 'query_images' and 'system_message').

        Returns:
            dict: A dictionary containing 'model_output', 'is_valid', 'response_time', 'n_output_tokens',
                and any other relevant information returned by the model. If 'chat_mode' is True, also
                includes 'previous_messages'.
        """
        response_dict = {}
        if hasattr(self, "system_message") and self.system_message:
            if "system_message" in kwargs:
                logging.warning(
                    "System message is passed via the dataloader but will be overridden by the model class system message."
                )
            kwargs["system_message"] = self.system_message
        request = self.create_request(query_text, *args, **kwargs)
        attempts = 0
        model_output = None
        is_valid = False
        response_time = None
        n_output_tokens = None

        while attempts < self.num_retries:
            try:
                model_response = self.get_response(request)
                if model_response:
                    response_dict.update(model_response)
                    model_output = model_response["model_output"]
                    response_time = model_response["response_time"]
                    n_output_tokens = model_response.get("n_output_tokens", None)
                if self.chat_mode:
                    previous_messages = self.update_chat_history(query_text, model_output, *args, **kwargs)

                is_valid = True
                break
            except Exception as e:
                logging.warning(f"Attempt {attempts+1}/{self.num_retries} failed: {e}")
                do_return = self.handle_request_error(e)
                if do_return:
                    break
                attempts += 1
        else:
            logging.warning("All attempts failed.")

        response_dict.update(
            {
                "is_valid": is_valid,
                "model_output": model_output,
                "response_time": response_time,
                "n_output_tokens": n_output_tokens or self.count_tokens(model_output, is_valid),
            }
        )
        if self.chat_mode:
            response_dict.update({"previous_messages": previous_messages})
        return response_dict

    @abstractmethod
    def handle_request_error(self, e):
        """Handle errors that occur during the request.

        Args:
            e (Exception): The exception encountered.

        Returns:
            bool: True if the error should cause an immediate return, False otherwise.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


@dataclass
class RestEndpointModel(EndpointModel, KeyBasedAuthMixIn):
    """Class for interacting with REST-based endpoint models.

    Inherits from:
        EndpointModel (Model)
        KeyBasedAuthMixIn

    Attributes:
        chat_mode (bool): Whether the model operates in chat mode (from Model).
        system_message (str): A system message that can be used by the model (from Model).
        num_retries (int): The number of attempts to retry the request (from EndpointModel).
        api_key (str): The API key used for authentication (from KeyBasedAuthMixIn).
        secret_key_params (dict): Parameters to retrieve the API key from secret manager (from KeyBasedAuthMixIn).
        url (str): The endpoint URL.
        model_name (str): The name of the model.
        temperature (float): The temperature for text generation.
        max_tokens (int): The maximum number of tokens to generate.
        top_p (float): The top-p sampling value.
        frequency_penalty (float): The frequency penalty parameter.
        presence_penalty (float): The presence penalty parameter.
        do_sample (bool): Whether to use sampling.
        timeout (int): The timeout in seconds for the request.
    """

    url: str = None
    model_name: str = None
    temperature: float = 0
    max_tokens: int = 2000
    top_p: float = 0.95
    frequency_penalty: float = 0
    presence_penalty: float = 0
    do_sample: bool = True
    timeout: int = None

    def create_request(self, text_prompt, query_images=None, system_message=None, previous_messages=None):
        """Create a request for the REST endpoint model.

        Args:
            text_prompt (str): The text prompt to generate the model response.
            query_images (list, optional): List of images in base64 bytes format to be included in the request.
                Defaults to None, as images are not supported.
            system_message (str, optional): System message to be included in the request. Defaults to None.
            previous_messages (list, optional): List of previous messages for maintaining chat state. Defaults to None.

        Returns:
            urllib.request.Request: The request object to be sent to the model endpoint.

        Raises:
            NotImplementedError: If images are provided, as they're not supported yet.
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if previous_messages:
            messages.extend(previous_messages)
        messages.append({"role": "user", "content": text_prompt})
        data = {
            "input_data": {
                "input_string": messages,
                "parameters": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "do_sample": self.do_sample,
                    "max_new_tokens": self.max_tokens,
                },
            }
        }
        if query_images:
            raise NotImplementedError("Images are not supported for RestEndpointModel endpoints yet.")

        body = str.encode(json.dumps(data))
        headers = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self.api_key),
        }

        return urllib.request.Request(self.url, body, headers)

    def get_response(self, request):
        """Obtain the model response from the REST endpoint.

        Measures the response time, parses the model output, and returns both.

        Args:
            request (urllib.request.Request): The request object to be sent to the model endpoint.

        Returns:
            dict: Contains 'model_output' and 'response_time'.
        """
        start_time = time.time()
        response = urllib.request.urlopen(request, timeout=self.timeout)
        end_time = time.time()
        res = json.loads(response.read())
        model_output = res["output"]
        response_time = end_time - start_time
        return {"model_output": model_output, "response_time": response_time}

    def handle_request_error(self, e):
        """Handle request errors for the REST endpoint.

        Logs detailed information about the error.

        Args:
            e (Exception): The exception encountered.

        Returns:
            bool: False to indicate that the operation should not proceed further.
        """
        if isinstance(e, urllib.error.HTTPError):
            logging.info("The request failed with status code: " + str(e.code))
            logging.info(e.info())
            logging.info(e.read().decode("utf8", "ignore"))
        else:
            logging.info("The request failed with: " + str(e))
        return False


@dataclass
class ServerlessAzureRestEndpointModel(EndpointModel, KeyBasedAuthMixIn):
    """Class for serverless Azure model deployments.

    Inherits from:
        EndpointModel (Model)
        KeyBasedAuthMixIn

    Additional information:
        https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-serverless?tabs=azure-ai-studio

    Attributes:
        chat_mode (bool): Whether the model operates in chat mode (from Model).
        system_message (str): A system message that can be used by the model (from Model).
        num_retries (int): The number of attempts to retry the request (from EndpointModel).
        api_key (str): The API key used for authentication (from KeyBasedAuthMixIn).
        secret_key_params (dict): Parameters to retrieve the API key from secret manager (from KeyBasedAuthMixIn).
        url (str): The endpoint URL.
        model_name (str): The name of the model.
        stream (bool): Whether or not to stream the response.
        auth_scope (str): The authentication scope for Azure.
        timeout (int): The timeout in seconds for the request.
    """

    url: str = None
    model_name: str = None
    stream: bool = False
    auth_scope: str = "https://cognitiveservices.azure.com/.default"
    timeout: int = None

    def __post_init__(self):
        """Initialize the serverless Azure REST endpoint model.

        Tries to initialize with either an API key or a bearer token provider. If neither is provided,
        obtains a bearer token from Azure CLI.
        """
        try:
            super().__post_init__()
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": ("Bearer " + self.api_key),
                "extra-parameters": "pass-through",
            }
        except ValueError:
            self.bearer_token_provider = get_bearer_token_provider(AzureCliCredential(), self.auth_scope)
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": ("Bearer " + self.bearer_token_provider()),
                "extra-parameters": "pass-through",
            }

    @abstractmethod
    def create_request(self, text_prompt, query_images=None, system_message=None, previous_messages=None):
        """Create a request for the serverless Azure REST endpoint model.

        Args:
            text_prompt (str): The text prompt for the model.
            query_images (list, optional): List of images to be included in the request. Defaults to None.
            system_message (str, optional): The system message for the request. Defaults to None.
            previous_messages (list, optional): Previous messages in the chat. Defaults to None.

        Returns:
            Any: The request object to be sent to the model.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_response(self, request):
        """Get the response from the Azure REST endpoint.

        Measures the response time, parses the model output, and includes usage details if present.

        Args:
            request (urllib.request.Request): The request object.

        Returns:
            dict: Contains 'model_output', 'response_time', and optionally 'usage'.
        """
        start_time = time.time()
        response = urllib.request.urlopen(request, timeout=self.timeout)
        end_time = time.time()
        res = json.loads(response.read())
        model_output = res["choices"][0]["message"]["content"]
        response_time = end_time - start_time
        response_dict = {
            "model_output": model_output,
            "response_time": response_time,
        }
        if "usage" in res:
            response_dict.update({"usage": res["usage"]})
        return response_dict

    def handle_request_error(self, e):
        """Handle errors during a serverless Azure REST endpoint request.

        Logs the error details for debugging.

        Args:
            e (Exception): The exception encountered.

        Returns:
            bool: False to indicate that the operation should not proceed further.
        """
        if isinstance(e, urllib.error.HTTPError):
            logging.info("The request failed with status code: " + str(e.code))
            logging.info(e.info())
            logging.info(e.read().decode("utf8", "ignore"))
        else:
            logging.info("The request failed with: " + str(e))
        return False


@dataclass
class LlamaServerlessAzureRestEndpointModel(ServerlessAzureRestEndpointModel):
    """Serverless Azure REST endpoint model for Llama-based deployments.

    Inherits from:
        ServerlessAzureRestEndpointModel (EndpointModel, KeyBasedAuthMixIn)

    Tested for Llama 3.1 405B Instruct deployments and Llama 3.2 90B Vision Instruct.
    See:
    https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-llama?tabs=llama-three

    Attributes:
        chat_mode (bool): Whether the model operates in chat mode (from Model).
        system_message (str): A system message that can be used by the model (from Model).
        num_retries (int): The number of attempts to retry (from EndpointModel).
        api_key (str): The API key (from KeyBasedAuthMixIn).
        secret_key_params (dict): Secret key params (from KeyBasedAuthMixIn).
        url (str): The endpoint URL (from ServerlessAzureRestEndpointModel).
        model_name (str): The name of the model.
        stream (bool): Whether to stream the response (from ServerlessAzureRestEndpointModel).
        auth_scope (str): The authentication scope (from ServerlessAzureRestEndpointModel).
        timeout (int): The timeout in seconds (from ServerlessAzureRestEndpointModel).
        temperature (float): The sampling temperature.
        max_tokens (int): The maximum number of tokens to generate.
        top_p (float): The top-p sampling value.
        frequency_penalty (float): The frequency penalty parameter.
        presence_penalty (float): The presence penalty parameter.
        use_beam_search (bool): Whether or not to use beam search.
        best_of (int): The best-of parameter for multiple sampling runs.
        skip_special_tokens (bool): Whether or not to skip special tokens in the output.
        ignore_eos (bool): Whether or not to ignore the EOS token.
    """

    temperature: float = 0
    max_tokens: int = 2000
    top_p: float = 0.95
    frequency_penalty: float = 0
    presence_penalty: float = 0
    use_beam_search: bool = False
    best_of: int = 1
    skip_special_tokens: bool = False
    ignore_eos: bool = False

    def create_request(self, text_prompt, query_images=None, system_message=None, previous_messages=None):
        """Create a request for the Llama serverless Azure REST endpoint.

        Supports an optional single image for the Llama vision model. The image is base64-encoded
        and included in the request if provided.

        Args:
            text_prompt (str): The user prompt text.
            query_images (list, optional): List containing a single image (for Llama vision model). Defaults to None.
            system_message (str, optional): The system message. Defaults to None.
            previous_messages (list, optional): Previous messages for maintaining chat history. Defaults to None.

        Returns:
            urllib.request.Request: The request object to be sent to the Llama endpoint.

        Raises:
            ValueError: If more than one image is provided for the Llama vision model.
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if previous_messages:
            messages.extend(previous_messages)
        user_content = text_prompt
        if query_images:
            if len(query_images) > 1:
                raise ValueError("Llama vision model does not support more than 1 image.")
            encoded_images = self.base64encode(query_images)
            user_content = [
                {"type": "text", "text": text_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_images[0]}",
                    },
                },
            ]
        messages.append({"role": "user", "content": user_content})

        data = {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "best_of": self.best_of,
            "presence_penalty": self.presence_penalty,
            "use_beam_search": self.use_beam_search,
            "ignore_eos": self.ignore_eos,
            "skip_special_tokens": self.skip_special_tokens,
            "stream": self.stream,
        }
        body = str.encode(json.dumps(data))
        return urllib.request.Request(self.url, body, self.headers)


@dataclass
class MistralServerlessAzureRestEndpointModel(ServerlessAzureRestEndpointModel):
    """Serverless Azure REST endpoint model for Mistral-based deployments.

    Inherits from:
        ServerlessAzureRestEndpointModel (EndpointModel, KeyBasedAuthMixIn)

    Tested for Mistral Large 2 2407 deployments. Refer to:
    https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-mistral?tabs=mistral-large#mistral-chat-api

    Attributes:
        chat_mode (bool): Whether the model operates in chat mode (from Model).
        system_message (str): A system message that can be used by the model (from Model).
        num_retries (int): The number of attempts to retry (from EndpointModel).
        api_key (str): The API key (from KeyBasedAuthMixIn).
        secret_key_params (dict): Secret key params (from KeyBasedAuthMixIn).
        url (str): The endpoint URL (from ServerlessAzureRestEndpointModel).
        model_name (str): The name of the model.
        stream (bool): Whether to stream the response (from ServerlessAzureRestEndpointModel).
        auth_scope (str): The authentication scope (from ServerlessAzureRestEndpointModel).
        timeout (int): The timeout in seconds (from ServerlessAzureRestEndpointModel).
        temperature (float): The sampling temperature.
        max_tokens (int): The maximum number of tokens to generate.
        top_p (float): The top-p sampling value.
        safe_prompt (bool): Whether or not to use a safe prompt.
    """

    temperature: float = 0
    max_tokens: int = 2000
    top_p: float = 1
    safe_prompt: bool = False

    def __post_init__(self):
        """Initialize the Mistral serverless Azure REST endpoint model.

        Enforces constraints on top_p if temperature is zero, following the Mistral chat API guidelines.
        """
        if self.temperature == 0 and self.top_p != 1:
            warning_message = (
                "Top_p must be 1 when using greedy sampling. Temperature zero means greedy sampling. "
                "Top_p will be reset to 1. See https://learn.microsoft.com/en-us/azure/ai-studio/"
                "how-to/deploy-models-mistral?tabs=mistral-large#mistral-chat-api for more information."
            )
            logging.warning(warning_message)
            self.top_p = 1
        super().__post_init__()

    def create_request(self, text_prompt, query_images=None, system_message=None, previous_messages=None):
        """Create a request for the Mistral serverless Azure REST endpoint.

        Args:
            text_prompt (str): The text prompt for which the model generates a response.
            query_images (list, optional): Not supported for this model. Defaults to None.
            system_message (str, optional): The system message. Defaults to None.
            previous_messages (list, optional): The previous messages in the chat. Defaults to None.

        Returns:
            urllib.request.Request: The request object.

        Raises:
            NotImplementedError: If images are provided, as they're not supported.
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if previous_messages:
            messages.extend(previous_messages)
        if query_images:
            raise NotImplementedError(
                "Images are not supported for MistralServerlessAzureRestEndpointModel endpoints."
            )
        messages.append({"role": "user", "content": text_prompt})
        data = {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "safe_prompt": self.safe_prompt,
            "stream": self.stream,
        }
        body = str.encode(json.dumps(data))
        return urllib.request.Request(self.url, body, self.headers)


@dataclass
class DeepseekR1ServerlessAzureRestEndpointModel(ServerlessAzureRestEndpointModel):
    """Serverless Azure REST endpoint model for DeepSeek-R1.

    Inherits from:
        ServerlessAzureRestEndpointModel (EndpointModel, KeyBasedAuthMixIn)

    See https://huggingface.co/deepseek-ai/DeepSeek-R1 for more details.

    Attributes:
        chat_mode (bool): Whether the model operates in chat mode (from Model).
        system_message (str): A system message that can be used by the model (from Model).
        num_retries (int): The number of attempts to retry (from EndpointModel).
        api_key (str): The API key (from KeyBasedAuthMixIn).
        secret_key_params (dict): Secret key params (from KeyBasedAuthMixIn).
        url (str): The endpoint URL (from ServerlessAzureRestEndpointModel).
        model_name (str): The name of the model.
        stream (bool): Whether to stream the response (from ServerlessAzureRestEndpointModel).
        auth_scope (str): The authentication scope (from ServerlessAzureRestEndpointModel).
        timeout (int): The timeout in seconds (from ServerlessAzureRestEndpointModel).
        temperature (float): The sampling temperature. Defaults to 0.6.
        max_tokens (int): The maximum number of tokens to generate. Defaults to 4096.
        top_p (float): The top-p sampling value. Defaults to 0.95.
        presence_penalty (float): The presence penalty parameter. Defaults to 0.
    """

    temperature: float = 0.6
    max_tokens: int = 4096
    top_p: float = 0.95
    presence_penalty: float = 0

    def create_request(self, text_prompt, query_images=None, system_message=None, previous_messages=None):
        """Create a request for the DeepSeek-R1 serverless Azure REST endpoint.

        Args:
            text_prompt (str): The text prompt to generate the model response.
            query_images (list, optional): Not supported for this model. Defaults to None.
            system_message (str, optional): The system message. Defaults to None.
            previous_messages (list, optional): The previous messages in the chat. Defaults to None.

        Returns:
            urllib.request.Request: The request object.

        Raises:
            NotImplementedError: If images are provided, as they're not supported.
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if previous_messages:
            messages.extend(previous_messages)
        if query_images:
            raise NotImplementedError(
                "Images are not supported for DeepseekR1ServerlessAzureRestEndpointModel endpoints."
            )
        messages.append({"role": "user", "content": text_prompt})
        data = {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
        }
        body = str.encode(json.dumps(data))
        return urllib.request.Request(self.url, body, self.headers)
    


@dataclass
class OpenAICommonRequestResponseMixIn:
    """Define request and response handling for most OpenAI models.

    This mixin provides methods to create a chat request body and parse the
    response from the OpenAI API.
    """

    def create_request(self, text_prompt, query_images=None, system_message=None, previous_messages=None):
        """Create a request dictionary for use with the OpenAI chat API.

        Args:
            text_prompt (str): The user-provided text prompt.
            query_images (Optional[List[str]], optional): A list of images to encode and include in the request, if any.
            system_message (Optional[str], optional): The system message to include in the conversation, if any.
            previous_messages (Optional[List[Dict]], optional): A list of previous messages in the conversation.

        Returns:
            dict: A request body dictionary that can be passed to the OpenAI API.
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if previous_messages:
            messages.extend(previous_messages)
        user_content = text_prompt
        if query_images:
            encoded_images = self.base64encode(query_images)
            user_content = [
                {"type": "text", "text": text_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_images[0]}",
                    },
                },
            ]
        messages.append({"role": "user", "content": user_content})
        request_body = {"messages": messages}
        for kwarg in {"extra_body"}:
            if hasattr(self, kwarg):
                request_body[kwarg] = getattr(self, kwarg)
        return request_body

    def get_response(self, request):
        """Send a chat completion request to the OpenAI API and return the parsed response.

        Args:
            request (dict): The request body to send to the OpenAI API.

        Returns:
            dict: A dictionary containing the model output, response time, and optional usage information.
        """
        start_time = time.time()
        completion = self.client.chat.completions.create(
            model=self.model_name,
            top_p=self.top_p,
            seed=self.seed,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **request,
        )
        end_time = time.time()
        openai_response = completion.model_dump()
        model_output = openai_response["choices"][0]["message"]["content"]
        response_time = end_time - start_time
        response_dict = {
            "model_output": model_output,
            "response_time": response_time,
        }
        if "usage" in openai_response:
            usage = openai_response["usage"]
            response_dict.update({"usage": usage})
            if isinstance(usage, dict) and "completion_tokens" in usage:
                response_dict.update({"n_output_tokens": usage["completion_tokens"]})
        return response_dict

    def base64encode(self, images):
        """Encode images in Base64 format.

        Args:
            images (List[str]): A list of image file paths.

        Returns:
            List[str]: A list of Base64-encoded strings corresponding to the images.
        """
        import base64

        encoded_list = []
        for img_path in images:
            with open(img_path, "rb") as image_file:
                encoded_list.append(base64.b64encode(image_file.read()).decode("utf-8"))
        return encoded_list


class AzureOpenAIClientMixIn:
    """Provide Azure OpenAI-specific client methods and error handling.
    """

    def get_client(self):
        """Retrieve the Azure OpenAI client.

        Returns:
            AzureOpenAI: The Azure OpenAI client instance.
        """
        from openai import AzureOpenAI

        token_provider = get_bearer_token_provider(AzureCliCredential(), self.auth_scope)
        return AzureOpenAI(
            azure_endpoint=self.url,
            api_version=self.api_version,
            azure_ad_token_provider=token_provider,
        )

    def handle_request_error(self, e):
        """Handle an error that occurs while making a request to the Azure OpenAI service.

        If the error is due to content filtering, this method logs a warning and
        returns (None, False, True). Otherwise, it logs the exception and returns False.

        Args:
            e (Exception): The exception raised during the request.

        Returns:
            Union[Tuple[None, bool, bool], bool]: Either a tuple indicating a content
            filter block or False indicating the request can be retried.
        """
        # if the error is due to a content filter, there is no need to retry
        if hasattr(e, "code") and e.code == "content_filter":
            logging.warning("Content filtered.")
            response = None
            return response, False, True
        else:
            logging.warning(str(e))
        return False


class DirectOpenAIClientMixIn(KeyBasedAuthMixIn):
    """Provide client retrieval and error handling for direct OpenAI usage.
    """

    def get_client(self):
        """Retrieve the direct OpenAI client.

        Returns:
            OpenAI: The direct OpenAI client instance.
        """
        from openai import OpenAI

        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def handle_request_error(self, e):
        """Handle an error that occurs while making a request to the direct OpenAI service.

        Args:
            e (Exception): The exception that was raised.

        Returns:
            bool: Always returns False, indicating the request can be retried.
        """
        logging.warning(e)
        return False


@dataclass
class AzureOpenAIModel(OpenAICommonRequestResponseMixIn, AzureOpenAIClientMixIn, EndpointModel):
    """Interact with Azure OpenAI models using the provided configuration.

    Attributes:
        url (str): The Azure OpenAI endpoint URL.
        model_name (str): The name of the model.
        temperature (float): The temperature setting for the model.
        max_tokens (int): The maximum number of tokens to generate.
        top_p (float): The top-p sampling parameter.
        frequency_penalty (float): The frequency penalty.
        presence_penalty (float): The presence penalty.
        seed (int): An optional seed for reproducibility.
        api_version (str): The API version used for Azure OpenAI.
        auth_scope (str): The authentication scope for Azure OpenAI.
        chat_mode (bool): Whether the model operates in chat mode (from Model).
        system_message (str): A system message that can be used by the model (from Model).
        num_retries (int): The number of attempts to retry the request (from EndpointModel).
    """

    url: str = None
    model_name: str = None
    temperature: float = 0
    max_tokens: int = 2000
    top_p: float = 0.95
    frequency_penalty: float = 0
    presence_penalty: float = 0
    seed: int = 0
    api_version: str = "2023-06-01-preview"
    auth_scope: str = "https://cognitiveservices.azure.com/.default"

    def __post_init__(self):
        """Initialize the AzureOpenAIModel instance by obtaining the Azure OpenAI client."""
        self.client = self.get_client()


@dataclass
class DirectOpenAIModel(OpenAICommonRequestResponseMixIn, DirectOpenAIClientMixIn, EndpointModel):
    """Interact directly with OpenAI models using the provided configuration.

    Attributes:
        model_name (str): The name of the model.
        temperature (float): The temperature setting for the model.
        max_tokens (int): The maximum number of tokens to generate.
        top_p (float): The top-p sampling parameter.
        frequency_penalty (float): The frequency penalty.
        presence_penalty (float): The presence penalty.
        seed (int): An optional seed for reproducibility.
        api_version (str): The API version used by OpenAI.
        base_url (str): The base URL for the OpenAI API.
        extra_body (dict): Additional data to include in the request body.
        chat_mode (bool): Whether the model operates in chat mode (from Model).
        system_message (str): A system message that can be used by the model (from Model).
        num_retries (int): The number of attempts to retry the request (from EndpointModel).
    """

    model_name: str = None
    temperature: float = 0
    max_tokens: int = 2000
    top_p: float = 0.95
    frequency_penalty: float = 0
    presence_penalty: float = 0
    seed: int = 0
    api_version: str = "2023-06-01-preview"
    base_url: str = "https://api.openai.com/v1"
    extra_body: dict = None

    def __post_init__(self):
        """Initialize the DirectOpenAIModel instance by obtaining the API key and direct OpenAI client."""
        self.api_key = self.get_api_key()
        self.client = self.get_client()


class OpenAIOModelsRequestResponseMixIn:
    """Define request creation and response handling for OpenAI O1 models.
    """

    def create_request(self, text_prompt, query_images=None, system_message=None, previous_messages=None):
        """Create a request dictionary for use with OpenAI O1 chat models.

        Args:
            text_prompt (str): The text prompt to send to the model.
            query_images (Optional[List[str]], optional): A list of images to encode and include, if supported.
            system_message (Optional[str], optional): A system or developer message to pass to the model.
            previous_messages (Optional[List[Dict]], optional): A list of previous conversation messages.

        Returns:
            dict: The request body dictionary to pass to the OpenAI API.
        """
        messages = []
        if system_message and "o1-preview" in self.model_name:
            logging.warning("System and developer messages are not supported by OpenAI O1 preview model.")
        elif system_message:
            # Developer messages are the new system messages
            messages.append({"role": "developer", "content": system_message})
        if previous_messages:
            messages.extend(previous_messages)

        user_content = text_prompt
        if query_images and "o1-preview" in self.model_name:
            logging.warning("Images are not supported by OpenAI O1 preview model.")
        elif query_images:
            encoded_images = self.base64encode(query_images)
            user_content = [
                {"type": "text", "text": text_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_images[0]}",
                    },
                },
            ]

        messages.append({"role": "user", "content": user_content})
        request_body = {"messages": messages}
        for kwarg in {"extra_body"}:
            if hasattr(self, kwarg):
                request_body[kwarg] = getattr(self, kwarg)
        return request_body

    def get_response(self, request):
        """Send the request to the OpenAI O1 model and return the parsed response.

        Depending on whether the model is an O1 preview or not, it may or may not support
        certain parameters such as developer/system messages or reasoning effort.

        Args:
            request (dict): The request body for the chat completion.

        Returns:
            dict: A dictionary containing the model output, response time, and optional usage details.
        """
        start_time = time.time()
        if "o1-preview" in self.model_name:
            if self.reasoning_effort == "high":
                logging.error("Reasoning effort is not supported by OpenAI O1 preview model.")
            completion = self.client.chat.completions.create(
                model=self.model_name,
                seed=self.seed,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                **request,
            )
        else:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                seed=self.seed,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                reasoning_effort=self.reasoning_effort,
                **request,
            )
        end_time = time.time()
        openai_response = completion.model_dump()
        model_output = openai_response["choices"][0]["message"]["content"]
        response_time = end_time - start_time
        response_dict = {
            "model_output": model_output,
            "response_time": response_time,
        }
        if "usage" in openai_response:
            response_dict.update({"usage": openai_response["usage"]})
        return response_dict

    def base64encode(self, images):
        """Encode images in Base64 format.

        Args:
            images (List[str]): A list of image file paths.

        Returns:
            List[str]: A list of Base64-encoded strings corresponding to the images.
        """
        import base64

        encoded_list = []
        for img_path in images:
            with open(img_path, "rb") as image_file:
                encoded_list.append(base64.b64encode(image_file.read()).decode("utf-8"))
        return encoded_list


@dataclass
class DirectOpenAIOModel(OpenAIOModelsRequestResponseMixIn, DirectOpenAIClientMixIn, EndpointModel):
    """Interact directly with OpenAI O1 models using the provided configuration.

    Attributes:
        model_name (str): The name of the model.
        temperature (float): The temperature setting for generation.
        max_completion_tokens (int): Not currently used due to API constraints.
        top_p (float): The top-p sampling parameter.
        seed (int): An optional seed for reproducibility.
        frequency_penalty (float): The frequency penalty.
        presence_penalty (float): The presence penalty.
        reasoning_effort (str): The level of reasoning effort requested.
        base_url (str): The base URL for the OpenAI API.
        extra_body (dict): Additional data to include in the request.
        chat_mode (bool): Whether the model operates in chat mode (from Model).
        system_message (str): A system message that can be used by the model (from Model).
        num_retries (int): The number of attempts to retry the request (from EndpointModel).
    """

    model_name: str = None
    temperature: float = 1
    max_completion_tokens: int = 2000
    top_p: float = 1
    seed: int = 0
    frequency_penalty: float = 0
    presence_penalty: float = 0
    reasoning_effort: str = "medium"
    base_url: str = "https://api.openai.com/v1"
    extra_body: dict = None

    def __post_init__(self):
        """Initialize the DirectOpenAIOModel instance by obtaining the API key and direct OpenAI client."""
        self.api_key = self.get_api_key()
        self.client = self.get_client()


@dataclass
class AzureOpenAIOModel(OpenAIOModelsRequestResponseMixIn, AzureOpenAIClientMixIn, EndpointModel):
    """Interact with Azure OpenAI O1 models using the provided configuration.

    Attributes:
        url (str): The Azure endpoint for O1 models.
        model_name (str): The name of the O1 model.
        temperature (float): The temperature setting for generation.
        max_completion_tokens (int): Not currently used due to API constraints.
        top_p (float): The top-p sampling parameter.
        seed (int): An optional seed for reproducibility.
        frequency_penalty (float): The frequency penalty.
        presence_penalty (float): The presence penalty.
        reasoning_effort (str): The level of reasoning effort requested.
        api_version (str): The API version for Azure capabilities.
        auth_scope (str): The scope for Azure authentication.
        chat_mode (bool): Whether the model operates in chat mode (from Model).
        system_message (str): A system message that can be used by the model (from Model).
        num_retries (int): The number of attempts to retry the request (from EndpointModel).
    """

    url: str = None
    model_name: str = None
    temperature: float = 1
    max_completion_tokens: int = 64000
    top_p: float = 1
    seed: int = 0
    frequency_penalty: float = 0
    presence_penalty: float = 0
    reasoning_effort: str = "medium"
    api_version: str = "2023-06-01-preview"
    auth_scope: str = "https://cognitiveservices.azure.com/.default"

    def __post_init__(self):
        """Initialize the AzureOpenAIOModel instance by obtaining the Azure OpenAI client."""
        self.client = self.get_client()


@dataclass
class GeminiModel(EndpointModel, KeyBasedAuthMixIn):
    """Interact with Gemini models through the Python API.

    Attributes:
        timeout (int): The API request timeout in seconds.
        model_name (str): The name of the Gemini model.
        temperature (float): The temperature setting for generation.
        max_tokens (int): The maximum number of tokens to generate.
        top_p (float): The top-p sampling parameter.
        chat_mode (bool): Chat mode is not implemented yet for Gemini models. (from Model).
        system_message (str): A system message that can be used by the model (from Model).
        num_retries (int): The number of attempts to retry the request (from EndpointModel).
    """

    timeout: int = 600
    model_name: str = None
    temperature: float = 0
    max_tokens: int = 2000
    top_p: float = 0.95

    def __post_init__(self):
        """Initialize the GeminiModel by configuring the generative AI client with the provided API key
        and safety settings. Also set up the generation config.
        """
        super().__post_init__()
        if self.chat_mode:
            logging.error("Chat mode is not implemented yet for Gemini models. Set chat_mode=False in your model config.")
        
        import google.generativeai as genai
        from google.generativeai.types import HarmBlockThreshold, HarmCategory

        genai.configure(api_key=self.api_key)
        # Safety config, turning off all filters for direct experimentation with the model only
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        self.gen_config = genai.GenerationConfig(
            max_output_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p
        )

    def create_request(self, text_prompt, query_images=None, system_message=None, previous_messages=None):
        """Create a request for generating content with Gemini models.

        Args:
            text_prompt (str): The text prompt to send to the model.
            query_images (Optional[List[str]], optional): Image data to pass to the model.
            system_message (Optional[str], optional): An optional system instruction to pass to the model.
            previous_messages (Optional[List[Dict]], optional): A list of previous conversation messages (unused).

        Returns:
            Union[str, List[str]]: The prompt alone, or a list combining the prompt and images.
        """
        import google.generativeai as genai

        if self.model_name == "gemini-1.0-pro":
            if system_message:
                logging.warning("System messages are not supported for Gemini 1.0 Pro.")
            self.model = genai.GenerativeModel(self.model_name)
        else:
            self.model = genai.GenerativeModel(self.model_name, system_instruction=system_message)

        if query_images:
            return [text_prompt] + query_images
        else:
            return text_prompt

    def get_response(self, request):
        """Send the request to the Gemini model and return the parsed response.

        Args:
            request (Union[str, List[str]]): The text prompt or a combined prompt and images.

        Returns:
            dict: A dictionary containing the model output, response time, and usage metadata if available.
        """
        start_time = time.time()
        gemini_response = None
        try:
            gemini_response = self.model.generate_content(
                request,
                generation_config=self.gen_config,
                request_options={"timeout": self.timeout},
                safety_settings=self.safety_settings,
            )
            end_time = time.time()
            model_output = gemini_response.parts[0].text
            response_time = end_time - start_time
        except Exception as e:
            self.handle_gemini_error(e, gemini_response)

        response_dict = {
            "model_output": model_output,
            "response_time": response_time,
        }
        if hasattr(gemini_response, "usage_metadata"):
            try:
                response_dict.update(
                    {
                        "usage": {
                            "prompt_token_count": gemini_response.usage_metadata.prompt_token_count,
                            "candidates_token_count": gemini_response.usage_metadata.candidates_token_count,
                            "total_token_count": gemini_response.usage_metadata.total_token_count,
                        }
                    }
                )
            except AttributeError:
                logging.warning("Usage metadata not found in the response.")
        return response_dict

    def handle_gemini_error(self, e, gemini_response):
        """Handle exceptions originating from making requests to the Gemini API.

        If the model explicitly or implicitly blocks the prompt, logs a warning indicating
        the block reason and re-raises the exception.

        Args:
            e (Exception): The exception that occurred.
            gemini_response (Any): The response object from the Gemini model.

        Raises:
            Exception: Re-raises the provided exception after handling.
        """
        # Handling cases where the model explicitly blocks prompts and provides a reason for it.
        if e.__class__.__name__ == "ValueError" and gemini_response.prompt_feedback.block_reason > 0:
            logging.warning(
                f"Attempt failed due to explicitly blocked input prompt: {e} Block Reason {gemini_response.prompt_feedback.block_reason}"
            )

        # Handling cases where the model implicitly blocks prompts and does not provide an explicit block reason.
        elif e.__class__.__name__ == "IndexError" and len(gemini_response.parts) == 0:
            logging.warning(f"Attempt failed due to implicitly blocked input prompt and empty model output: {e}")
            if len(gemini_response.candidates) > 0:
                logging.warning(f"The response is not empty and has : {len(gemini_response.candidates)} candidates")
                logging.warning(
                    f"Finish Reason for the first answer candidate is: {gemini_response.candidates[0].finish_reason}"
                )
                logging.warning(
                    f"Safety Ratings for the first answer candidate are: {gemini_response.candidates[0].safety_ratings}"
                )

        raise e

    def handle_request_error(self, e):
        """Handle an error that occurs while making a request to the Gemini model.

        Any error case not handled in handle_gemini_error will be attempted again.

        Args:
            e (Exception): The exception that was raised.

        Returns:
            bool: Always returns False, indicating the request can be retried.
        """
        return False


@dataclass
class TogetherModel(OpenAICommonRequestResponseMixIn, KeyBasedAuthMixIn, EndpointModel):
    """Interact with Together models through the together Python API.

    Attributes:
        timeout (int): The API request timeout in seconds.
        model_name (str): The name of the Together model.
        temperature (float): The temperature setting for generation.
        max_tokens (int): The maximum number of tokens to generate.
        top_p (float): The top-p sampling parameter.
        presence_penalty (float): The presence penalty.
        stop (List[str]): A list of stop tokens for generation.
        chat_mode (bool): Whether the model operates in chat mode (from Model).
        system_message (str): A system message that can be used by the model (from Model).
        num_retries (int): The number of attempts to retry the request (from EndpointModel).
        
    """

    timeout: int = 600
    model_name: str = None
    temperature: float = 0
    max_tokens: int = 65536
    top_p: float = 0.95
    presence_penalty: float = 0
    stop = ["<｜end▁of▁sentence｜>"]

    def __post_init__(self):
        """Initialize the TogetherModel by setting up the Together client with the provided API key."""
        from together import Together

        self.api_key = self.get_api_key()
        self.client = Together(api_key=self.api_key)

    def get_response(self, request):
        """Send the request to the Together model and return the parsed response.

        Args:
            request (dict): The request body for the Together chat completion.

        Returns:
            dict: A dictionary containing the model output, response time, and optional usage details.
        """
        start_time = time.time()
        completion = self.client.chat.completions.create(
            model=self.model_name,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=self.stop,
            **request,
        )

        end_time = time.time()
        openai_response = completion.model_dump()
        model_output = openai_response["choices"][0]["message"]["content"]
        response_time = end_time - start_time
        response_dict = {
            "model_output": model_output,
            "response_time": response_time,
        }
        if "usage" in openai_response:
            response_dict.update({"usage": openai_response["usage"]})
        return response_dict

    def handle_request_error(self, e):
        """Handle an error that occurs while making a request to the Together service.

        Args:
            e (Exception): The exception that was raised.

        Returns:
            bool: Always returns False, indicating the request can be retried.
        """
        return False

@dataclass
class HuggingFaceModel(Model):
    """
    Runs a self-hosted language model via HuggingFace APIs.

    This class handles loading and running a HuggingFace language model locally
    with optional quantization and flash attention usage.

    Attributes:
        model_name (str): The name of the HuggingFace model to use.
        device (str): The device to use for inference (e.g., 'cpu' or 'cuda').
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature for sampling-based generation.
        top_p (float): The top-p (nucleus) sampling parameter.
        do_sample (bool): Whether to sample or not. Setting to False uses greedy
            decoding.
        apply_model_template (bool): If True, applies a template to the prompt
            before generating.
        quantize (bool): Whether to quantize the model for memory savings.
        use_flash_attn (bool): Whether to use flash attention 2 if supported.
        chat_mode (bool): Not used. (from Model).
        system_message (str): Not used. (from Model).
    """

    model_name: str = None
    device: str = "cpu"
    max_tokens: int = 2000
    temperature: float = 0.001
    top_p: float = 0.95
    do_sample: bool = True
    apply_model_template: bool = True

    quantize: bool = False
    use_flash_attn: bool = False

    def __post_init__(self):
        """
        Initializes model-related attributes.

        Sets the device by picking an available GPU or falling back to CPU
        before loading the model.
        """
        # The device needs to be set before get_model() is called
        self.device = self.pick_available_device()
        self.get_model()

    def get_model(self):
        """
        Loads the HuggingFace model and tokenizer.

        If quantization is enabled, applies 4-bit quantization. Otherwise, loads
        the model with standard precision. The tokenizer is also loaded using
        the same model name.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quantization_config = None
        if self.quantize:
            from transformers import BitsAndBytesConfig

            logging.info("Quantizing model")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            device_map=self.device,
            use_flash_attention_2=self.use_flash_attn,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

    def pick_available_device(self):
        """
        Selects the device with the lowest GPU utilization, or defaults to CPU.

        Returns:
            str: The name of the chosen device (e.g., 'cuda:0' or 'cpu').
        """
        import numpy as np
        import torch

        device = "cpu"

        if torch.cuda.is_available():
            utilizations = []
            for i in range(torch.cuda.device_count()):
                util = torch.cuda.utilization(f"cuda:{i}")
                utilizations.append(util)

            gpu_index = np.argmin(utilizations)

            device = f"cuda:{gpu_index}"

        logging.info(f"Using device {device} for model self hosting")

        return device

    def _generate(self, text_prompt, query_images=None):
        """
        Generates text given a prompt and optional images.

        Args:
            text_prompt (str): The text prompt for the model to process.
            query_images (list, optional): A list of images (if supported by the model).

        Returns:
            dict: A dictionary with the following keys:
                model_output (str): The generated text response.
                response_time (float): The time taken for the generation.
        """
        import time

        inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)
        start_time = time.time()
        output_ids = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
        )
        end_time = time.time()
        sequence_length = inputs["input_ids"].shape[1]
        new_output_ids = output_ids[:, sequence_length:]
        model_output = self.tokenizer.batch_decode(
            new_output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        response_time = end_time - start_time
        return {
            "model_output": model_output,
            "response_time": response_time,
        }

    def generate(self, text_prompt, query_images=None, system_message=None):
        """
        Generates a text response, optionally applying a model-specific template.

        Args:
            text_prompt (str): The text prompt for generation.
            query_images (list, optional): A list of images (if supported by the model).
            system_message (str, optional): A system message to be prepended or otherwise
                integrated into the template.

        Returns:
            dict: A dictionary containing the following keys:
                model_output (str): The generated text response.
                response_time (float): The time taken for generation.
                is_valid (bool): Whether the generation was successful.
                n_output_tokens (int): The number of tokens in the generated output.
        """
        response_dict = {}

        if text_prompt:
            if self.apply_model_template:
                text_prompt = self.model_template_fn(text_prompt, system_message)

            try:
                model_response = self._generate(text_prompt, query_images=query_images)
                if model_response:
                    response_dict.update(model_response)
                is_valid = True

            except Exception as e:
                logging.warning(e)
                is_valid = False

        response_dict.update(
            {
                "is_valid": is_valid,
                "n_output_tokens": self.count_tokens(response_dict["model_output"], response_dict["is_valid"]),
            }
        )
        return response_dict

    def model_template_fn(self, text_prompt, system_message=None):
        """
        Applies a basic template to the prompt.

        If a system message is provided, it is prepended to the text prompt.

        Args:
            text_prompt (str): The original text prompt.
            system_message (str, optional): A system message to prepend.

        Returns:
            str: The prompt with the optional system message attached.
        """
        return system_message + " " + text_prompt if system_message else text_prompt


@dataclass
class Phi3HFModel(HuggingFaceModel):
    """
    Runs a self-hosted PHI3 model via HuggingFace APIs.

    This class extends HuggingFaceModel, applying a PHI3-specific prompt template.
    """

    def __post_init__(self):
        """
        Initializes and checks if the model name is a PHI3 model.

        Warns if the provided model name is not a PHI3 model.
        """
        super().__post_init__()
        if "microsoft/Phi-3" not in self.model_name:
            logging.warning(
                "This model class applies a template to the prompt that is specific to Phi-3 models"
                " but your model is not a Phi-3 model."
            )

    def model_template_fn(self, text_prompt, system_message=None):
        """
        Applies the PHI3-specific template to the prompt.

        Args:
            text_prompt (str): The text prompt for generation.
            system_message (str, optional): A system message to prepend.

        Returns:
            str: The prompt with the PHI3-specific template.
        """
        text_prompt = super().model_template_fn(text_prompt, system_message)
        return f"<|user|>\n{text_prompt}<|end|>\n<|assistant|>"


@dataclass
class Phi4HFModel(HuggingFaceModel):
    """
    Runs a self-hosted PHI4 model via HuggingFace APIs.

    This class extends HuggingFaceModel, applying a PHI4-specific prompt template.
    """

    def __post_init__(self):
        """
        Initializes and checks if the model name is a PHI4 model.

        Warns if the provided model name is not a PHI4 model.
        """
        super().__post_init__()
        if "microsoft/phi-4" not in self.model_name:
            logging.warning(
                "This model class applies a template to the prompt that is specific to Phi-4 models"
                " but your model is not a Phi-4 model."
            )

    def model_template_fn(self, text_prompt, system_message=None):
        """
        Applies the PHI4-specific template to the prompt.

        Args:
            text_prompt (str): The text prompt for generation.
            system_message (str, optional): A system message to prepend.

        Returns:
            str: The prompt with the PHI4-specific template.
        """
        if system_message:
            return f"<|im_start|>system<|im_sep|>\n{system_message}<|im_start|>user<|im_sep|>\n{text_prompt}<|im_end|>\n<|im_start|>assistant<|im_sep|>"
        else:
            return f"<|im_start|>user<|im_sep|>\n{text_prompt}<|im_end|>\n<|im_start|>assistant<|im_sep|>"


@dataclass
class LLaVAHuggingFaceModel(HuggingFaceModel):
    """
    Runs a self-hosted LLaVA model via HuggingFace APIs.

    This class extends HuggingFaceModel, applying an image-based prompt template
    for LLaVA.
    """

    def __post_init__(self):
        """
        Initializes and checks if the model name is a LLaVA model.

        Warns if the provided model name is not a LLaVA model.
        """
        super().__post_init__()
        if "llava" not in self.model_name:
            logging.warning(
                "This model class applies a template to the prompt that is specific to LLAVA models"
                " but your model is not a LLAVA model."
            )

    def get_model(self):
        """
        Loads the LLaVA model and processor.

        If quantization is enabled, applies 4-bit quantization. Otherwise, loads
        the model with standard precision. The appropriate LLaVA variant is
        chosen depending on the model name (v1.6, etc.).
        """
        import torch
        from transformers import (
            AutoProcessor,
            LlavaForConditionalGeneration,
            LlavaNextForConditionalGeneration,
            LlavaNextProcessor,
        )

        quantization_config = None
        if self.quantize:
            from transformers import BitsAndBytesConfig

            logging.info("Quantizing model")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        if "v1.6" in self.model_name:
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                device_map=self.device,
                use_flash_attention_2=self.use_flash_attn,
            )
            self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                device_map=self.device,
                use_flash_attention_2=self.use_flash_attn,
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)

    def _generate(self, text_prompt, query_images=None):
        """
        Generates a response given a text prompt and optional images.

        Args:
            text_prompt (str): The text prompt for generation.
            query_images (list, optional): A list containing a single image to
                use as context.

        Returns:
            dict: A dictionary containing the following keys:
                model_output (str): The generated text response.
                response_time (float): The time taken for the generation.
        """
        import time

        inputs = self.processor(text=text_prompt, images=query_images, return_tensors="pt").to(self.device)
        start_time = time.time()
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
        )
        end_time = time.time()
        sequence_length = inputs["input_ids"].shape[1]
        new_output_ids = output_ids[:, sequence_length:]
        model_output = self.processor.batch_decode(
            new_output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        response_time = end_time - start_time
        return {
            "model_output": model_output,
            "response_time": response_time,
        }

    def generate(self, text_prompt, query_images=None, system_message=None):
        """
        Generates a response using a LLaVA model, with optional images.

        Args:
            text_prompt (str): The text prompt for generation.
            query_images (list, optional): A list containing a single image to
                use.
            system_message (str, optional): An additional system message or
                instruction.

        Returns:
            dict: A dictionary containing the generation results. If multiple
                images are provided, returns an error message in the dictionary.
        """
        if query_images and len(query_images) > 1:
            logging.error(f"Not implemented for more than 1 image. {len(query_images)} images are in the prompt")
            return {"model_output": None, "is_valid": False, "response_time": None, "n_output_tokens": None}

        return super().generate(text_prompt, query_images=query_images, system_message=system_message)

    def model_template_fn(self, text_prompt, system_message=None):
        """
        Applies an image-based template to the text prompt for LLaVA models.

        The exact template depends on the LLaVA model variant (v1.6, v1.6-mistral,
        etc.).

        Args:
            text_prompt (str): The text prompt to transform.
            system_message (str, optional): An additional system message.

        Returns:
            str: The transformed text prompt with an image token included.
        """
        text_prompt = f"<image>\n{text_prompt}"

        if "v1.6-mistral" in self.model_name:
            text_prompt = f"[INST] {text_prompt} [/INST]"
        elif "v1.6-vicuna" in self.model_name:
            if system_message:
                text_prompt = f"{system_message} USER: {text_prompt} ASSISTANT:"
            else:
                text_prompt = (
                    f"A chat between a curious human and an artificial intelligence assistant. "
                    f"The assistant gives helpful, detailed, and polite answers to the human's questions. "
                    f"USER: {text_prompt} ASSISTANT:"
                )
        elif "v1.6-34b" in self.model_name:
            if system_message:
                text_prompt = f"<|im_start|>system\n{system_message}<|im_end|><|im_start|>user\n{text_prompt}<|im_end|><|im_start|>assistant\n"
            else:
                text_prompt = f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n{text_prompt}<|im_end|><|im_start|>assistant\n"
        else:
            if system_message:
                text_prompt = f"{system_message} USER: {text_prompt} ASSISTANT:"
            else:
                text_prompt = f"USER: {text_prompt} ASSISTANT:"

        return text_prompt


@dataclass
class LLaVAModel(LLaVAHuggingFaceModel):
    """
    Runs a self-hosted LLaVA model using the LLaVA package.

    This class extends LLaVAHuggingFaceModel to handle model loading and
    inference through the dedicated LLaVA package utilities.

    """

    model_base: str = None
    num_beams: int = 1

    def __post_init__(self):
        """
        Initializes the LLaVA model after the dataclass has been populated.
        """
        super().__post_init__()

    def get_model(self):
        """
        Loads the LLaVA model and tokenizer using the LLaVA package.

        This method overrides the base HuggingFace loading routine and uses
        dedicated LLaVA utilities for model setup.
        """
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        disable_torch_init()

        self.model_path = self.model_name
        self.model_name = get_model_name_from_path(self.model_path)

        tokenizer, model, processor, _ = load_pretrained_model(
            self.model_path,
            self.model_base,
            self.model_name,
            load_4bit=self.quantize,
            device_map="auto",
            device=self.device,
            use_flash_attn=self.use_flash_attn,
        )
        model.eval()

        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

    def _generate(self, text_prompt, query_images=None, system_message=None):
        """
        Generates a response using the LLaVA package.

        Args:
            text_prompt (str): The text prompt to process.
            query_images (list, optional): A list of images for multimodal generation.
            system_message (str, optional): An additional system message for context.

        Returns:
            dict: A dictionary containing the following keys:
                model_output (str): The generated text response.
                response_time (float): The time taken for generation.
        """
        import torch
        import time
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import process_images, tokenizer_image_token

        image_sizes = [x.size for x in query_images]

        images_tensor = process_images(query_images, self.processor, self.model.config).to(
            self.device, dtype=torch.float16
        )

        input_ids = (
            tokenizer_image_token(text_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.inference_mode():
            start_time = time.time()
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=self.do_sample,
                temperature=self.temperature,
                num_beams=self.num_beams,
                max_new_tokens=self.max_tokens,
                use_cache=True,
            )
            end_time = time.time()

        model_output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        response_time = end_time - start_time
        return {
            "model_output": model_output,
            "response_time": response_time,
        }

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
        chat_mode (bool): Not used. (from Model).
        system_message (str): Not used. (from Model).
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
        """Initialize the vLLM model after dataclass fields are set.

        Returns:
            None
        """
        # vLLM automatically picks an available devices when get_model() is called
        self.get_model()

    def get_model(self):
        """Initialize and store the LLM instance from vLLM.

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
        """Generate a response from the model using the vLLM chat interface.

        Args:
            text_prompt (str): The prompt for the model.
            query_images (Any, optional): Additional images to pass to the model. Defaults to None.

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
        """Generate a response from the model.

        Args:
            text_prompt (str): The prompt for the model.
            query_images (Any, optional): Additional images to pass to the model. Defaults to None.
            system_message (str, optional): A system message to prepend. Defaults to None.

        Returns:
            dict: A dictionary with the generated response. The dictionary includes
                "model_output", "is_valid", "response_time", and "n_output_tokens".
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
        """Create a list of messages suitable for the vLLM chat interface.

        Args:
            text_prompt (str): The user's prompt.
            system_message (str, optional): A system message to prepend.

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
    """Handle the deployment of vLLM servers.

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
        """Initialize the local vLLM deployment handler.

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
        """Get clients to access vLLM servers.

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
        """Check if vLLM servers are running.

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
        """Deploy vLLM servers in background threads using the specified parameters.

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
        """Deploy a single vLLM server.

        Uses gpus_per_port GPUs starting at index * gpus_per_port.

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
        """Shut down all vLLM servers deployed during this run.

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
    """Represent a local vLLM server deployment.

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
        num_retries (int): Number of retries for failed requests (from EndpointModel).
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
        """Initialize the local vLLM model deployment.

        Raises:
            ValueError: If model_name is not specified.
        """
        if not self.model_name:
            raise ValueError("LocalVLLM model_name must be specified.")
        self.handler = self._get_local_vllm_deployment_handler()

    @property
    def client(self):
        """Get a randomly selected client from the list of deployed servers.

        Returns:
            OpenAIClient: A client for sending requests to the vLLM server.
        """
        return random.choice(self.handler.clients)

    def _get_local_vllm_deployment_handler(self):
        """Get or create a local vLLM deployment handler for this model.

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
        """Handle any request error that occurs during inference.

        Args:
            e (Exception): The exception raised during inference.

        Returns:
            bool: Always returns False indicating the request was not successful.
        """
        return False


@dataclass
class ClaudeModel(EndpointModel, KeyBasedAuthMixIn):
    """Interact with Claude models through the Python API.

    Attributes:
        model_name (str): Name of the Claude model.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum number of tokens in the generated response.
        top_p (float): Nucleus sampling probability threshold.
        timeout (int): Request timeout in seconds.
        chat_mode (bool): Whether the model operates in chat mode (from Model).
        system_message (str): A system message that can be used by the model (from Model).
        num_retries (int): The number of attempts to retry the request (from EndpointModel).

    """

    model_name: str = None
    temperature: float = 0
    max_tokens: int = 2000
    top_p: float = 0.95
    timeout: int = 60

    def __post_init__(self):
        """Initialize the Claude client after dataclass fields are set.

        Returns:
            None
        """
        super().__post_init__()
        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def create_request(self, text_prompt, query_images=None, system_message=None, previous_messages=None):
        """Create a request payload for Claude.

        Args:
            text_prompt (str): The user's prompt.
            query_images (Any, optional): Additional images to encode and send. Defaults to None.
            system_message (str, optional): A system message to prepend. Defaults to None.
            previous_messages (list, optional): A list of previous conversation messages. Defaults to None.

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
        """Send a request to the Claude model and get a response.

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
        """Handle any request error that occurs during Claude inference.

        Args:
            e (Exception): The exception raised during inference.

        Returns:
            bool: Always returns False indicating the request was not successful.
        """
        return False


@dataclass
class ClaudeReasoningModel(ClaudeModel):
    """Interact with Claude reasoning models through the Python API.

    Allows usage of a 'thinking' parameter for advanced reasoning.

    Attributes:
        model_name (str): Name of the Claude reasoning model.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum number of tokens in the generated response.
        timeout (int): Request timeout in seconds.
        thinking_enabled (bool): Whether thinking mode is enabled.
        thinking_budget (int): Token budget for thinking mode.
        top_p (float): This parameter is not supported and will be ignored.
        chat_mode (bool): Whether the model operates in chat mode (from Model).
        system_message (str): A system message that can be used by the model (from Model).
        num_retries (int): The number of attempts to retry the request (from EndpointModel).
    """

    model_name: str = None
    temperature: float = 1.0
    max_tokens: int = 20000
    timeout: int = 600
    thinking_enabled: bool = True
    thinking_budget: int = 16000
    top_p: float = None

    def get_response(self, request):
        """Send a request to the Claude reasoning model and get a response.

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
        """Generate a test response.

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