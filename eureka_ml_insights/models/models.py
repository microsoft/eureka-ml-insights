"""This module contains classes for interacting with various models, including API-based models and HuggingFace models."""

import json
import logging
import random
import requests
import threading
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass

import anthropic
import tiktoken
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from eureka_ml_insights.secret_management import get_secret


@dataclass
class Model(ABC):
    """This class is used to define the structure of a model class.
    Any model class should inherit from this class and implement the generate method that returns a dict
    containing the model_output, is_valid, and other relevant information.
    """

    chat_mode: bool = False

    @abstractmethod
    def generate(self, text_prompt, *args, **kwargs):
        raise NotImplementedError

    def count_tokens(self, model_output: str = None, is_valid: bool = False):
        """
        This method uses tiktoken tokenizer to count the number of tokens in the response.
        See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        args:
            model_output (str): the text response from the model.
            is_valid (bool): whether the response is valid or not.
        returns:
            n_output_tokens (int): the number of tokens in the text response.
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        if model_output is None or not is_valid:
            return None
        else:
            n_output_tokens = len(encoding.encode(model_output))
            return n_output_tokens

    def base64encode(self, query_images):
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
    """This class is used to handle key-based authentication for models."""

    api_key: str = None
    secret_key_params: dict = None

    def __post_init__(self):
        if self.api_key is None and self.secret_key_params is None:
            raise ValueError("Either api_key or secret_key_params must be provided.")
        self.api_key = self.get_api_key()

    def get_api_key(self):
        """
        This method is used to get the api_key for the models that require key-based authentication.
        Either api_key (str) or secret_key_params (dict) must be provided.
        if api_key is not directly provided, secret_key_params must be provided to get the api_key using get_secret method.
        """
        if self.api_key is None:
            self.api_key = get_secret(**self.secret_key_params)
        return self.api_key


@dataclass
class EndpointModel(Model):
    """This class is used to interact with API-based models."""

    num_retries: int = 3

    @abstractmethod
    def create_request(self, text_prompt, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_response(self, request):
        # must return the model output and the response time
        raise NotImplementedError

    def update_chat_history(self, query_text, model_output, *args, **kwargs):
        """
        This method is used to update the chat history with the model response.
        args:
            query_text (str): the text prompt to generate the response.
            model_output (str): the text response from the model.
        returns:
            previous_messages (list): a list of messages in the chat history.
        """
        previous_messages = kwargs.get("previous_messages", [])
        previous_messages.append({"role": "user", "content": query_text})
        previous_messages.append({"role": "assistant", "content": model_output})
        return previous_messages

    def generate(self, query_text, *args, **kwargs):
        """
        Calls the endpoint to generate the model response.
        args:
            query_text (str): the text prompt to generate the response.
            query_images (list): list of images in base64 bytes format to be included in the request.
            system_message (str): the system message to be included in the request.
        returns:
            response_dict (dict): a dictionary containing the model_output, is_valid, response_time, and n_output_tokens,
                                  and any other relevant information returned by the model.
        """
        response_dict = {}
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
        raise NotImplementedError


@dataclass
class RestEndpointModel(EndpointModel, KeyBasedAuthMixIn):
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
        """Creates a request for the model."""
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
        # The azureml-model-deployment header will force the request to go to a specific deployment.
        # Remove this header to have the request observe the endpoint traffic rules
        headers = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self.api_key),
        }

        return urllib.request.Request(self.url, body, headers)

    def get_response(self, request):
        # Get the model response and measure the time taken.
        start_time = time.time()
        response = urllib.request.urlopen(request, timeout=self.timeout)
        end_time = time.time()
        # Parse the response and return the model output.
        res = json.loads(response.read())
        model_output = res["output"]
        response_time = end_time - start_time
        return {"model_output": model_output, "response_time": response_time}

    def handle_request_error(self, e):
        if isinstance(e, urllib.error.HTTPError):
            logging.info("The request failed with status code: " + str(e.code))
            # Print the headers - they include the request ID and the timestamp, which are useful for debugging.
            logging.info(e.info())
            logging.info(e.read().decode("utf8", "ignore"))
        else:
            logging.info("The request failed with: " + str(e))
        return False


@dataclass
class ServerlessAzureRestEndpointModel(EndpointModel, KeyBasedAuthMixIn):
    """This class can be used for serverless Azure model deployments."""

    """https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-serverless?tabs=azure-ai-studio"""
    url: str = None
    model_name: str = None
    stream: bool = False
    auth_scope: str = "https://cognitiveservices.azure.com/.default"
    timeout: int = None

    def __post_init__(self):
        try:
            super().__post_init__()
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": ("Bearer " + self.api_key),
                # The behavior of the API when extra parameters are indicated in the payload.
                # Using pass-through makes the API to pass the parameter to the underlying model.
                # Use this value when you want to pass parameters that you know the underlying model can support.
                # https://learn.microsoft.com/en-us/azure/machine-learning/reference-model-inference-chat-completions?view=azureml-api-2
                "extra-parameters": "pass-through",
            }
        except ValueError:
            self.bearer_token_provider = get_bearer_token_provider(DefaultAzureCredential(), self.auth_scope)
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": ("Bearer " + self.bearer_token_provider()),
                # The behavior of the API when extra parameters are indicated in the payload.
                # Using pass-through makes the API to pass the parameter to the underlying model.
                # Use this value when you want to pass parameters that you know the underlying model can support.
                # https://learn.microsoft.com/en-us/azure/machine-learning/reference-model-inference-chat-completions?view=azureml-api-2
                "extra-parameters": "pass-through",
            }

    @abstractmethod
    def create_request(self, text_prompt, query_images=None, system_message=None, previous_messages=None):
        # Exact model parameters are model-specific.
        # The method cannot be implemented unless the model being deployed is known.
        raise NotImplementedError

    def get_response(self, request):
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
        if isinstance(e, urllib.error.HTTPError):
            logging.info("The request failed with status code: " + str(e.code))
            # Print the headers - they include the request ID and the timestamp, which are useful for debugging.
            logging.info(e.info())
            logging.info(e.read().decode("utf8", "ignore"))
        else:
            logging.info("The request failed with: " + str(e))
        return False


@dataclass
class LlamaServerlessAzureRestEndpointModel(ServerlessAzureRestEndpointModel):
    """Tested for Llama 3.1 405B Instruct deployments and Llama 3.2 90B Vision Instruct."""

    """See https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-llama?tabs=llama-three for the api reference."""

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
    """Tested for Mistral Large 2 2407 deployments."""

    """See https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-mistral?tabs=mistral-large#mistral-chat-api for the api reference."""
    temperature: float = 0
    max_tokens: int = 2000
    top_p: float = 1
    safe_prompt: bool = False

    def __post_init__(self):
        if self.temperature == 0 and self.top_p != 1:
            warning_message = "Top_p must be 1 when using greedy sampling. Temperature zero means greedy sampling. Top_p will be reset to 1. See https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-mistral?tabs=mistral-large#mistral-chat-api for more information."
            logging.warning(warning_message)
            self.top_p = 1
        super().__post_init__()

    def create_request(self, text_prompt, query_images=None, system_message=None, previous_messages=None):
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
            # Safe_prompt activates an optional system prompt to enforce guardrails.
            # See https://docs.mistral.ai/capabilities/guardrailing/
            "safe_prompt": self.safe_prompt,
            "stream": self.stream,
        }
        body = str.encode(json.dumps(data))
        return urllib.request.Request(self.url, body, self.headers)


@dataclass
class DeepseekR1ServerlessAzureRestEndpointModel(ServerlessAzureRestEndpointModel):
    # setting temperature to 0.6 as suggested in https://huggingface.co/deepseek-ai/DeepSeek-R1
    temperature: float = 0.6
    max_tokens: int = 4096
    top_p: float = 0.95
    presence_penalty: float = 0

    def create_request(self, text_prompt, query_images=None, system_message=None, previous_messages=None):
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
    """
    This mixin class defines the request and response handling for most OpenAI models.
    """

    def create_request(self, text_prompt, query_images=None, system_message=None, previous_messages=None):
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
        return {"messages": messages}

    def get_response(self, request):
        start_time = time.time()
        completion = self.client.chat.completions.create(
            model=self.model_name,
            top_p=self.top_p,
            # seed=self.seed,
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


class AzureOpenAIClientMixIn:
    """This mixin provides some methods to interact with Azure OpenAI models."""

    def get_client(self):
        from openai import AzureOpenAI

        token_provider = get_bearer_token_provider(DefaultAzureCredential(), self.auth_scope)
        return AzureOpenAI(
            azure_endpoint=self.url,
            api_version=self.api_version,
            azure_ad_token_provider=token_provider,
        )

    def handle_request_error(self, e):
        # if the error is due to a content filter, there is no need to retry
        if hasattr(e, "code") and e.code == "content_filter":
            logging.warning("Content filtered.")
            response = None
            return response, False, True
        else:
            logging.warning(str(e))
        return False


class DirectOpenAIClientMixIn(KeyBasedAuthMixIn):
    """This mixin class provides some methods for using OpenAI models dirctly (not through Azure)"""

    def get_client(self):
        from openai import OpenAI

        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def handle_request_error(self, e):
        logging.warning(e)
        return False


@dataclass
class AzureOpenAIModel(OpenAICommonRequestResponseMixIn, AzureOpenAIClientMixIn, EndpointModel):
    """This class is used to interact with Azure OpenAI models."""

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
        self.client = self.get_client()


@dataclass
class DirectOpenAIModel(OpenAICommonRequestResponseMixIn, DirectOpenAIClientMixIn, EndpointModel):
    """This class is used to interact with OpenAI models dirctly (not through Azure)"""

    model_name: str = None
    temperature: float = 0
    max_tokens: int = 2000
    top_p: float = 0.95
    frequency_penalty: float = 0
    presence_penalty: float = 0
    seed: int = 0
    api_version: str = "2023-06-01-preview"
    base_url: str = "https://api.openai.com/v1"

    def __post_init__(self):
        self.api_key = self.get_api_key()
        self.client = self.get_client()


class OpenAIOModelsRequestResponseMixIn:
    def create_request(self, text_prompt, query_images=None, system_message=None, previous_messages=None):
        messages = []
        if system_message and "o1-preview" in self.model_name:
            logging.warning("System and developer messages are not supported by OpenAI O1 preview model.")
        elif system_message:
            # Developer messages are the new system messages:
            # Starting with o1-2024-12-17, o1 models support developer messages rather than system messages,
            # to align with the chain of command behavior described in the model spec.
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
        return {"messages": messages}

    def get_response(self, request):
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


@dataclass
class DirectOpenAIOModel(OpenAIOModelsRequestResponseMixIn, DirectOpenAIClientMixIn, EndpointModel):
    model_name: str = None
    temperature: float = 1
    # Not used currently, because the API throws:
    # "Completions.create() got an unexpected keyword argument 'max_completion_tokens'"
    # although this argument is documented in the OpenAI API documentation.
    max_completion_tokens: int = 2000
    top_p: float = 1
    seed: int = 0
    frequency_penalty: float = 0
    presence_penalty: float = 0
    reasoning_effort: str = "medium"
    base_url: str = "https://api.openai.com/v1"

    def __post_init__(self):
        self.api_key = self.get_api_key()
        self.client = self.get_client()


@dataclass
class AzureOpenAIOModel(OpenAIOModelsRequestResponseMixIn, AzureOpenAIClientMixIn, EndpointModel):
    url: str = None
    model_name: str = None
    temperature: float = 1
    # Not used currently, because the API throws:
    # "Completions.create() got an unexpected keyword argument 'max_completion_tokens'"
    # although this argument is documented in the OpenAI API documentation.
    max_completion_tokens: int = 2000
    top_p: float = 1
    seed: int = 0
    frequency_penalty: float = 0
    presence_penalty: float = 0
    reasoning_effort: str = "medium"
    api_version: str = "2023-06-01-preview"
    auth_scope: str = "https://cognitiveservices.azure.com/.default"

    def __post_init__(self):
        self.client = self.get_client()


@dataclass
class GeminiModel(EndpointModel, KeyBasedAuthMixIn):
    """This class is used to interact with Gemini models through the python api."""

    timeout: int = 600
    model_name: str = None
    temperature: float = 0
    max_tokens: int = 2000
    top_p: float = 0.95

    def __post_init__(self):
        super().__post_init__()
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
        """Handles exceptions originating from making requests to Gemini through the python api.

        args:
            e: Exception that occurred during getting a response.
            gemini_response: The response object from the gemini model.
        returns:
            _type_: do_return (True if the call should not be attempted again).
        """
        # Handling cases where the model explicitly blocks prompts and provides a reason for it.
        # In these cases, there is no need to make a new attempt as the model will continue to explicitly block the request, do_return = True.
        if e.__class__.__name__ == "ValueError" and gemini_response.prompt_feedback.block_reason > 0:
            logging.warning(
                f"Attempt failed due to explicitly blocked input prompt: {e} Block Reason {gemini_response.prompt_feedback.block_reason}"
            )

        # Handling cases where the model implicitly blocks prompts and does not provide an explicit block reason for it but rather an empty content.
        # In these cases, there is no need to make a new attempt as the model will continue to implicitly block the request, do_return = True.
        # Note that, in some cases, the model may still provide a finish reason as shown here https://ai.google.dev/api/generate-content?authuser=2#FinishReason
        elif e.__class__.__name__ == "IndexError" and len(gemini_response.parts) == 0:
            logging.warning(f"Attempt failed due to implicitly blocked input prompt and empty model output: {e}")
            # For cases where there are some response candidates do_return is still True because in most cases these candidates are incomplete.
            # Trying again may not necessarily help, unless in high temperature regimes.
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
        # Any error case not handled in handle_gemini_error will be attempted again, do_return = False.
        return False


@dataclass
class TogetherModel(OpenAICommonRequestResponseMixIn, KeyBasedAuthMixIn, EndpointModel):
    """This class is used to interact with Together models through the together python api."""

    timeout: int = 600
    model_name: str = None
    temperature: float = 0
    max_tokens: int = 65536
    top_p: float = 0.95
    presence_penalty: float = 0
    stop = ["<｜end▁of▁sentence｜>"]

    def __post_init__(self):
        from together import Together

        self.api_key = self.get_api_key()
        self.client = Together(api_key=self.api_key)

    def get_response(self, request):
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
        return False


@dataclass
class HuggingFaceModel(Model):
    """This class is used to run a self-hosted language model via HuggingFace apis."""

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
        # The device need to be set before get_model() is called
        self.device = self.pick_available_device()
        self.get_model()

    def get_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quantization_config = None
        if self.quantize:
            from transformers import BitsAndBytesConfig

            logging.info("Quantizing model")
            # specify how to quantize the model
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
        This method will enumerate all GPU devices and return the one with the lowest utilization.
        This is useful in running locally hosted HuggingFace models on multi-gpu machines.
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
        return system_message + " " + text_prompt if system_message else text_prompt


@dataclass
class Phi3HFModel(HuggingFaceModel):
    """This class is used to run a self-hosted PHI3 model via HuggingFace apis."""

    def __post_init__(self):
        super().__post_init__()
        if "microsoft/Phi-3" not in self.model_name:
            logging.warning(
                "This model class applies a template to the prompt that is specific to Phi-3 models"
                "but your model is not a Phi-3 model."
            )

    def model_template_fn(self, text_prompt, system_message=None):
        text_prompt = super().model_template_fn(text_prompt, system_message)
        return f"<|user|>\n{text_prompt}<|end|>\n<|assistant|>"


@dataclass
class Phi4HFModel(HuggingFaceModel):
    """This class is used to run a self-hosted PHI3 model via HuggingFace apis."""

    def __post_init__(self):
        super().__post_init__()
        if "microsoft/phi-4" not in self.model_name:
            logging.warning(
                "This model class applies a template to the prompt that is specific to Phi-4 models"
                "but your model is not a Phi-4 model."
            )

    def model_template_fn(self, text_prompt, system_message=None):
        if system_message:
            return f"<|im_start|>system<|im_sep|>\n{system_message}<|im_start|>user<|im_sep|>\n{text_prompt}<|im_end|>\n<|im_start|>assistant<|im_sep|>"
        else:
            return f"<|im_start|>user<|im_sep|>\n{text_prompt}<|im_end|>\n<|im_start|>assistant<|im_sep|>"


@dataclass
class LLaVAHuggingFaceModel(HuggingFaceModel):
    """This class is used to run a self-hosted LLaVA model via HuggingFace apis."""

    def __post_init__(self):
        super().__post_init__()
        if "llava" not in self.model_name:
            logging.warning(
                "This model class applies a template to the prompt that is specific to LLAVA models"
                "but your model is not a LLAVA model."
            )

    def get_model(self):
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
            # specify how to quantize the model
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

        if query_images and len(query_images) > 1:
            logging.error(f"Not implemented for more than 1 image. {len(query_images)} images are in the prompt")
            return {"model_output": None, "is_valid": False, "response_time": None, "n_output_tokens": None}

        return super().generate(text_prompt, query_images=query_images, system_message=system_message)

    def model_template_fn(self, text_prompt, system_message=None):
        text_prompt = f"<image>\n{text_prompt}"

        if "v1.6-mistral" in self.model_name:
            text_prompt = f"[INST] {text_prompt} [/INST]"
        elif "v1.6-vicuna" in self.model_name:
            if system_message:
                text_prompt = f"{system_message} USER: {text_prompt} ASSISTANT:"
            else:
                text_prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: {text_prompt} ASSISTANT:"
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
    """This class is used to run a self-hosted LLaVA model via the LLaVA package."""

    model_base: str = None
    num_beams: int = 1

    def __post_init__(self):
        super().__post_init__()

    def get_model(self):
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

        import torch
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
    """This class is used to run a self-hosted language model via vLLM apis.
    This class uses the chat() functionality of vLLM which applies a template included in the HF model files.
    If the model files do not include a template, no template will be applied.
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
        # vLLM automatically picks an available devices when get_model() is called
        self.get_model()

    def get_model(self):
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
        if system_message:
            return [
                {"role": "system", "content": system_message},
                {"role": "user", "content": text_prompt},
            ]
        else:
            return [{"role": "user", "content": text_prompt}]
    

class _LocalVLLMDeploymentHandler:
    """This class is used to handle the deployment of vLLM servers."""
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

    def _get_clients(self):
        '''Get clients to access vllm servers, by checking for running servers and deploying if necessary.'''
        from openai import OpenAI as OpenAIClient

        # If the user passes ports, check if the servers are running and populate clients accordingly.
        if self.ports:
            healthy_server_urls = ['http://0.0.0.0:' + port + '/v1' for port in self.get_healthy_ports()]
            if len(healthy_server_urls) > 0:
                logging.info(f"Found {len(healthy_server_urls)} healthy servers.")
                return [OpenAIClient(base_url=url, api_key = 'none') for url in healthy_server_urls]

        # Even if the user doesn't pass ports, we can check if there happen to be deployed servers.
        # There is no guarantee that the servers are hosting the correct model.
        self.ports = [str(8000 + i) for i in range(self.num_servers)]
        healthy_server_urls = ['http://0.0.0.0:' + port + '/v1' for port in self.get_healthy_ports()]
        if len(healthy_server_urls) == self.num_servers:
            logging.info(f"Found {len(healthy_server_urls)} healthy servers.")
            return [OpenAIClient(base_url=url, api_key = 'none') for url in healthy_server_urls]
        
        # If that didn't work, let's deploy and wait for servers to come online.
        self.deploy_servers()
        server_start_time = time.time()
        while time.time() - server_start_time < 600:
            time.sleep(10)
            healthy_ports = self.get_healthy_ports()
            if len(healthy_ports) == self.num_servers:
                logging.info(f"All {self.num_servers} servers are online.")
                healthy_server_urls = ['http://0.0.0.0:' + port + '/v1' for port in healthy_ports]
                return [OpenAIClient(base_url=url, api_key = 'none') for url in healthy_server_urls]
            else:
                logging.info(f"Waiting for {self.num_servers - len(healthy_ports)} more servers to come online.")
        
        if len(self.clients) != self.num_servers:
            raise RuntimeError(f"Failed to start all servers.")

    def get_healthy_ports(self) -> list[str]:
        """Check if servers are running."""

        healthy_ports = []
        for port in self.ports:
            try:
                self.session.get('http://0.0.0.0:' + port +'/health')
                healthy_ports.append(port)
            except:
                pass
        return healthy_ports
    
    def deploy_servers(self):
        """Deploy vLLM servers in background threads using the specified parameters."""

        logging.info(f"No vLLM servers are running. Starting {self.num_servers} new servers at {self.ports}.")
        import os, datetime

        gpus_per_port = self.pipeline_parallel_size * self.tensor_parallel_size
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        log_dir = os.path.join("logs", "local_vllm_deployment_logs", f"{date}")
        os.makedirs(log_dir)

        for index in range(self.num_servers):
            port = 8000 + index
            log_file = os.path.join(log_dir, f"{port}.log")
            self.logs.append(log_file)
            background_thread = threading.Thread(
                target = lambda: self.deploy_server(index, gpus_per_port, log_file)
            )
            background_thread.daemon = True
            background_thread.start()

    def deploy_server(self, index: int, gpus_per_port: int, log_file: str):
        """Deploy a single vLLM server using gpus_per_port many gpus starting at index*gpus_per_port."""
        
        import subprocess
        port = 8000 + index
        first_gpu = index * gpus_per_port
        last_gpu = first_gpu + gpus_per_port - 1
        devices = ",".join(str(gpu_num) for gpu_num in range(first_gpu, last_gpu + 1))

        command = [
            "CUDA_VISIBLE_DEVICES=" + devices,
            "vllm serve",
            self.model_name,
            "--port", str(port),
            "--tensor_parallel_size", str(self.tensor_parallel_size),
            "--pipeline_parallel_size", str(self.pipeline_parallel_size),
            "--dtype", self.dtype,
            "--seed", str(self.seed),
            "--gpu_memory_utilization", str(self.gpu_memory_utilization),
            "--cpu_offload_gb", str(self.cpu_offload_gb)
        ]
        if self.quantization:
            command.append("--quantization")
            command.append(self.quantization)
        if self.trust_remote_code:
            command.append("--trust_remote_code")
        command = " ".join(command)
        logging.info(f"Running command: {command}")
        with open(log_file, 'w') as log_writer:
            subprocess.run(command, shell=True, stdout=log_writer, stderr=log_writer)

    @classmethod
    def shutdown_servers(cls):
        """Shutdown all vLLM servers deployed during this run."""

        import re, os, signal
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
local_vllm_deployment_handlers : dict[str, _LocalVLLMDeploymentHandler] = {}
    
        
@dataclass
class LocalVLLMModel(OpenAICommonRequestResponseMixIn, EndpointModel):
    """This class is used for vLLM servers running locally.
    
    In case the servers are already deployed, specify the
    model_name and the ports at which the servers are hosted.
    Otherwise instantiating will initiate a deployment with
    any deployment parameters specified."""

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
    top_p: float = .95
    top_k: int = -1
    max_tokens: int = 2000
    frequency_penalty: float = 0
    presence_penalty: float = 0

    def __post_init__(self):
        if not self.model_name:
            raise ValueError("LocalVLLM model_name must be specified.")
        self.handler = self._get_local_vllm_deployment_handler()

    @property
    def client(self):
        return random.choice(self.handler.clients)
        
    def _get_local_vllm_deployment_handler(self):
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
        return False


@dataclass
class ClaudeModel(EndpointModel, KeyBasedAuthMixIn):
    """This class is used to interact with Claude models through the python api."""

    model_name: str = None
    temperature: float = 0
    max_tokens: int = 2000
    top_p: float = 0.95
    timeout: int = 60

    def __post_init__(self):
        super().__post_init__()
        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def create_request(self, text_prompt, query_images=None, system_message=None, previous_messages=None):
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
        return False


@dataclass
class ClaudeReasoningModel(ClaudeModel):
    """This class is used to interact with Claude reasoning models through the python api."""

    model_name: str = None
    temperature: float = 1.0
    max_tokens: int = 20000
    timeout: int = 600
    thinking_enabled: bool = True
    thinking_budget: int = 16000
    top_p: float = None

    def get_response(self, request):
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
    # This class is used for testing purposes only. It only waits for a specified time and returns a response.

    def generate(self, text_prompt, **kwargs):
        output = "This is a test response."
        is_valid = True
        return {
            "model_output": output,
            "is_valid": is_valid,
            "response_time": 0.1,
            "n_output_tokens": self.count_tokens(output, is_valid),
        }
