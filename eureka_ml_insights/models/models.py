"""This module contains classes for interacting with various models, including API-based models and HuggingFace models."""

import json
import logging
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

    model_output: str = None
    is_valid: bool = False
    response_time: float = None
    n_output_tokens: int = None
    chat_mode: bool = False
    previous_messages: list = None

    @abstractmethod
    def generate(self, text_prompt, *args, **kwargs):
        raise NotImplementedError

    def count_tokens(self):
        """
        This method uses tiktoken tokenizer to count the number of tokens in the response.
        See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        returns:
            n_output_tokens (int): the number of tokens in the text response.
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        if self.model_output is None or not self.is_valid:
            return None
        else:
            n_output_tokens = len(encoding.encode(self.model_output))
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

    def update_chat_history(self, query_text, *args, **kwargs):
        """
        This method is used to update the chat history with the model response.
        args:
            query_text (str): the text prompt to generate the response.
        returns:
            previous_messages (list): a list of messages in the chat history.
        """
        previous_messages = kwargs.get("previous_messages", [])
        previous_messages.append({"role": "user", "content": query_text})
        previous_messages.append({"role": "assistant", "content": self.model_output})
        self.previous_messages = previous_messages

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
        while attempts < self.num_retries:
            try:
                meta_response = self.get_response(request)
                if self.chat_mode:
                    self.update_chat_history(query_text, *args, **kwargs)
                if meta_response:
                    response_dict.update(meta_response)
                self.is_valid = True
                break
            except Exception as e:
                logging.warning(f"Attempt {attempts+1}/{self.num_retries} failed: {e}")
                do_return = self.handle_request_error(e)
                if do_return:
                    self.model_output = None
                    self.is_valid = False
                    break
                attempts += 1
        else:
            logging.warning("All attempts failed.")
            self.is_valid = False
            self.model_output = None

        response_dict.update(
            {
                "is_valid": self.is_valid,
                "model_output": self.model_output,
                "response_time": self.response_time,
                "n_output_tokens": self.count_tokens(),
            }
        )
        if self.chat_mode:
            response_dict.update({"previous_messages": self.previous_messages})
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
        self.model_output = res["output"]
        self.response_time = end_time - start_time

    def handle_request_error(self, e):
        if isinstance(e, urllib.error.HTTPError):
            logging.info("The request failed with status code: " + str(e.code))
            # Print the headers - they include the request ID and the timestamp, which are useful for debugging.
            logging.info(e.info())
            logging.info(e.read().decode("utf8", "ignore"))
        else:
            logging.info("The request failed with: "+ str(e))
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
                "extra-parameters": "pass-through"
            }
        except ValueError:
            self.bearer_token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), self.auth_scope
            )
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": ("Bearer " + self.bearer_token_provider()),
                # The behavior of the API when extra parameters are indicated in the payload. 
                # Using pass-through makes the API to pass the parameter to the underlying model. 
                # Use this value when you want to pass parameters that you know the underlying model can support.
                # https://learn.microsoft.com/en-us/azure/machine-learning/reference-model-inference-chat-completions?view=azureml-api-2
                "extra-parameters": "pass-through"
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
        self.model_output = res["choices"][0]["message"]["content"]
        self.response_time = end_time - start_time
        if "usage" in res:
            return {"usage": res["usage"]}

    def handle_request_error(self, e):
        if isinstance(e, urllib.error.HTTPError):
            logging.info("The request failed with status code: " + str(e.code))
            # Print the headers - they include the request ID and the timestamp, which are useful for debugging.
            logging.info(e.info())
            logging.info(e.read().decode("utf8", "ignore"))
        else:
            logging.info("The request failed with: "+ str(e))
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
        self.model_output = openai_response["choices"][0]["message"]["content"]
        self.response_time = end_time - start_time
        if "usage" in openai_response:
            return {"usage": openai_response["usage"]}


class AzureOpenAIClientMixIn:
    """This mixin provides some methods to interact with Azure OpenAI models."""

    def get_client(self):
        from openai import AzureOpenAI

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), self.auth_scope
        )
        return AzureOpenAI(
            azure_endpoint=self.url,
            api_version=self.api_version,
            azure_ad_token_provider=token_provider,
        )

    def handle_request_error(self, e):
        # if the error is due to a content filter, there is no need to retry
        if e.code == "content_filter":
            logging.warning("Content filtered.")
            response = None
            return response, False, True
        return False


class DirectOpenAIClientMixIn(KeyBasedAuthMixIn):
    """This mixin class provides some methods for using OpenAI models dirctly (not through Azure)"""

    def get_client(self):
        from openai import OpenAI

        return OpenAI(
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

    def __post_init__(self):
        self.api_key = self.get_api_key()
        self.client = self.get_client()


class OpenAIO1RequestResponseMixIn:
    
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
        completion = self.client.chat.completions.create(
            model=self.model_name,
            seed=self.seed,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            **request,
        )
        end_time = time.time()
        openai_response = completion.model_dump()
        self.model_output = openai_response["choices"][0]["message"]["content"]
        self.response_time = end_time - start_time
        if "usage" in openai_response:
            return {"usage": openai_response["usage"]}


@dataclass
class DirectOpenAIO1Model(OpenAIO1RequestResponseMixIn, DirectOpenAIClientMixIn, EndpointModel):
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

    def __post_init__(self):
        self.api_key = self.get_api_key()
        self.client = self.get_client()


@dataclass
class AzureOpenAIO1Model(OpenAIO1RequestResponseMixIn, AzureOpenAIClientMixIn, EndpointModel):
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
        self.gemini_response = self.model.generate_content(
            request,
            generation_config=self.gen_config,
            request_options={"timeout": self.timeout},
            safety_settings=self.safety_settings,
        )
        end_time = time.time()
        self.model_output = self.gemini_response.parts[0].text
        self.response_time = end_time - start_time
        if hasattr(self.gemini_response, "usage_metadata"):
            try:
                return {
                    "usage": {
                        "prompt_token_count": self.gemini_response.usage_metadata.prompt_token_count,
                        "candidates_token_count": self.gemini_response.usage_metadata.candidates_token_count,
                        "total_token_count": self.gemini_response.usage_metadata.total_token_count,
                    }
                }
            except AttributeError:
                logging.warning("Usage metadata not found in the response.")

    def handle_request_error(self, e):
        """Handles exceptions originating from making requests to Gemini through the python api.

        args:
            e (_type_): Exception occurred during getting a response.

        returns:
            _type_: do_return (True if the call should not be attempted again).
        """
        # Handling cases where the model explicitly blocks prompts and provides a reason for it.
        # In these cases, there is no need to make a new attempt as the model will continue to explicitly block the request, do_return = True.
        if e.__class__.__name__ == "ValueError" and self.gemini_response.prompt_feedback.block_reason > 0:
            logging.warning(
                f"Attempt failed due to explicitly blocked input prompt: {e} Block Reason {self.gemini_response.prompt_feedback.block_reason}"
            )
            return True
        # Handling cases where the model implicitly blocks prompts and does not provide an explicit block reason for it but rather an empty content.
        # In these cases, there is no need to make a new attempt as the model will continue to implicitly block the request, do_return = True.
        # Note that, in some cases, the model may still provide a finish reason as shown here https://ai.google.dev/api/generate-content?authuser=2#FinishReason
        elif e.__class__.__name__ == "IndexError" and len(self.gemini_response.parts) == 0:
            logging.warning(f"Attempt failed due to implicitly blocked input prompt and empty model output: {e}")
            # For cases where there are some response candidates do_return is still True because in most cases these candidates are incomplete.
            # Trying again may not necessarily help, unless in high temperature regimes.
            if len(self.gemini_response.candidates) > 0:
                logging.warning(f"The response is not empty and has : {len(self.gemini_response.candidates)} candidates")
                logging.warning(f"Finish Reason for the first answer candidate is: {self.gemini_response.candidates[0].finish_reason}")
                logging.warning(f"Safety Ratings for the first answer candidate are: {self.gemini_response.candidates[0].safety_ratings}")
            return True
        # Any other case will be re attempted again, do_return = False.
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
    stop=["<｜end▁of▁sentence｜>"]

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
            stop = self.stop,
            **request,
        )
        
        end_time = time.time()
        openai_response = completion.model_dump()
        self.model_output = openai_response["choices"][0]["message"]["content"]
        self.response_time = end_time - start_time
        if "usage" in openai_response:
            return {"usage": openai_response["usage"]}

    def handle_request_error(self, e):
        logging.warning(e)
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
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

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
        self.model_output = self.tokenizer.batch_decode(
            new_output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        self.response_time = end_time - start_time

    def generate(self, text_prompt, query_images=None, system_message=None):
        response_dict = {}

        if text_prompt:
            if self.apply_model_template:
                text_prompt = self.model_template_fn(text_prompt, system_message)

            try:
                meta_response = self._generate(text_prompt, query_images=query_images)
                if meta_response:
                    response_dict.update(meta_response)
                self.is_valid = True

            except Exception as e:
                logging.warning(e)
                self.is_valid = False

        response_dict.update(
            {
                "model_output": self.model_output,
                "is_valid": self.is_valid,
                "response_time": self.response_time,
                "n_output_tokens": self.count_tokens(),
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
        self.model_output = self.processor.batch_decode(
            new_output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        self.response_time = end_time - start_time

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

        self.model_output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        self.response_time = end_time - start_time


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
        self.model_output = completion.content[0].text
        self.response_time = end_time - start_time
        if hasattr(completion, "usage"):
            return {"usage": completion.usage.to_dict()}

    def handle_request_error(self, e):
        return False


@dataclass
class TestModel(Model):
    # This class is used for testing purposes only. It only waits for a specified time and returns a response.
    response_time: float = 0.1
    model_output: str = "This is a test response."

    def __post_init__(self):
        self.n_output_tokens = self.count_tokens()

    def generate(self, text_prompt, **kwargs):
        return {
            "model_output": self.model_output,
            "is_valid": True,
            "response_time": self.response_time,
            "n_output_tokens": self.n_output_tokens,
        }
