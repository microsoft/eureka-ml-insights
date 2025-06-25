"""Module for interacting with various models, including API-based models and HuggingFace models.

This module provides multiple classes that define and manage model interactions, including
chat-based and endpoint-based workflows. They handle text-based requests, token counting,
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
    """Base class for defining the structure of a model.

    Any model class should inherit from this class and implement the 'generate' method that returns
    a dictionary containing the 'model_output', 'is_valid', and other relevant information.
    """

    chat_mode: bool = False
    system_message: str = None

    @abstractmethod
    def generate(self, text_prompt, *args, **kwargs):
        """Generates a response from the model.

        Args:
            text_prompt (str): The text prompt for which the model needs to generate a response.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    def count_tokens(self, model_output: str = None, is_valid: bool = False):
        """Counts the number of tokens in the response.

        Uses the tiktoken tokenizer to count the number of tokens in the model's response. See:
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

        Args:
            model_output (str, optional): The text response from the model.
            is_valid (bool, optional): Whether the response is valid.

        Returns:
            int or None: The number of tokens in the text response, or None if the response is invalid or None.
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        if model_output is None or not is_valid:
            return None
        else:
            n_output_tokens = len(encoding.encode(model_output))
            return n_output_tokens

    def base64encode(self, query_images):
        """Encodes a list of images as base64 strings.

        Args:
            query_images (list): A list of PIL Images to be encoded as base64 strings.

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
    """Mixin class that handles key-based authentication for models."""

    api_key: str = None
    secret_key_params: dict = None

    def __post_init__(self):
        """Initializes the mixin, ensuring that either an API key or secret key parameters are provided.

        Raises:
            ValueError: If neither an API key nor secret key parameters are provided.
        """
        if self.api_key is None and self.secret_key_params is None:
            raise ValueError("Either api_key or secret_key_params must be provided.")
        self.api_key = self.get_api_key()

    def get_api_key(self):
        """Gets the API key for key-based authentication.

        Either the 'api_key' or 'secret_key_params' must be provided. If 'api_key' is not directly provided,
        'secret_key_params' must be used to retrieve the API key using 'get_secret'.

        Returns:
            str: The API key.
        """
        if self.api_key is None:
            self.api_key = get_secret(**self.secret_key_params)
        return self.api_key


@dataclass
class EndpointModel(Model):
    """Abstract class for interacting with API-based models."""

    num_retries: int = 3

    @abstractmethod
    def create_request(self, text_prompt, *args, **kwargs):
        """Creates a request for the model.

        Args:
            text_prompt (str): The text prompt to generate the request.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def get_response(self, request):
        """Gets the response from the model.

        Args:
            request: The request object created by create_request.

        Returns:
            dict: A dictionary containing the model output and the response time.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def update_chat_history(self, query_text, model_output, *args, **kwargs):
        """Updates the chat history with the model response.

        Args:
            query_text (str): The text prompt to generate the model response.
            model_output (str): The text response from the model.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            list: A list of messages representing the chat history.
        """
        previous_messages = kwargs.get("previous_messages", [])
        previous_messages.append({"role": "user", "content": query_text})
        previous_messages.append({"role": "assistant", "content": model_output})
        return previous_messages

    def generate(self, query_text, *args, **kwargs):
        """Calls the endpoint to generate the model response.

        Args:
            query_text (str): The text prompt to generate the response.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments (may include 'query_images' and 'system_message').

        Returns:
            dict: A dictionary containing 'model_output', 'is_valid', 'response_time', 'n_output_tokens',
                and any other relevant information returned by the model.
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
        """Handles errors that occur during the request.

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
    """Class for interacting with REST-based endpoint models."""

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
        """Creates a request for the REST endpoint model.

        Args:
            text_prompt (str): The text prompt to generate the model response.
            query_images (list, optional): List of images in base64 bytes format to be included in the request.
            system_message (str, optional): System message to be included in the request.
            previous_messages (list, optional): List of previous messages to maintain chat state.

        Returns:
            urllib.request.Request: The request object to be sent to the model endpoint.

        Raises:
            NotImplementedError: If images are passed, as they're not supported yet.
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
        """Obtains the model response from the REST endpoint.

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
        """Handles request errors for the REST endpoint.

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

    Additional information:
    https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-serverless?tabs=azure-ai-studio
    """

    url: str = None
    model_name: str = None
    stream: bool = False
    auth_scope: str = "https://cognitiveservices.azure.com/.default"
    timeout: int = None

    def __post_init__(self):
        """Initializes the serverless Azure REST endpoint model with either an API key or a bearer token."""
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
        """Creates a request for the serverless Azure REST endpoint model.

        Args:
            text_prompt (str): The text prompt for the model.
            query_images (list, optional): List of images to be included in the request.
            system_message (str, optional): System message for the request.
            previous_messages (list, optional): Previous messages in the chat.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_response(self, request):
        """Gets the response from the Azure REST endpoint.

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
        """Handles errors that occur during a serverless Azure REST endpoint request.

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

    Tested for Llama 3.1 405B Instruct deployments and Llama 3.2 90B Vision Instruct.
    See:
    https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-llama?tabs=llama-three
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
        """Creates a request for the Llama serverless Azure REST endpoint.

        Supports an optional single image for the Llama vision model. The image is base64-encoded
        and added to the request message if provided.

        Args:
            text_prompt (str): The user prompt text.
            query_images (list, optional): List containing a single image (for Llama vision model).
            system_message (str, optional): The system message to be included.
            previous_messages (list, optional): Previous messages for maintaining chat history.

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

    Tested for Mistral Large 2 2407 deployments. Refer to:
    https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-mistral?tabs=mistral-large#mistral-chat-api
    """

    temperature: float = 0
    max_tokens: int = 2000
    top_p: float = 1
    safe_prompt: bool = False

    def __post_init__(self):
        """Initializes the Mistral serverless Azure REST endpoint model.

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
        """Creates a request for the Mistral serverless Azure REST endpoint.

        Args:
            text_prompt (str): The text prompt for which the model generates a response.
            query_images (list, optional): Not supported for this model.
            system_message (str, optional): The system message to be included.
            previous_messages (list, optional): Previous messages in the chat.

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
            raise NotImplementedError("Images are not supported for MistralServerlessAzureRestEndpointModel endpoints.")
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

    Parameters are set following suggestions from:
    https://huggingface.co/deepseek-ai/DeepSeek-R1
    """

    temperature: float = 0.6
    max_tokens: int = 4096
    top_p: float = 0.95
    presence_penalty: float = 0

    def create_request(self, text_prompt, query_images=None, system_message=None, previous_messages=None):
        """Creates a request for the DeepSeek-R1 serverless Azure REST endpoint.

        Args:
            text_prompt (str): The text prompt to generate the model response.
            query_images (list, optional): Not supported for this model.
            system_message (str, optional): The system message to be included in the request.
            previous_messages (list, optional): The previous messages in the chat.

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