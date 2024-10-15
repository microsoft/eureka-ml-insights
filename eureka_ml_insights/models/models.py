"""This module contains classes for interacting with various models, including API-based models and HuggingFace models."""

import json
import logging
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass

import anthropic
from azure.identity import AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider, DefaultAzureCredential

from ratelimit import rate_limited, sleep_and_retry

from eureka_ml_insights.data_utils import GetKey


@dataclass
class Model(ABC):
    """This class is used to define the structure of a model class."""

    @abstractmethod
    def generate(self, text_prompt, query_images=None):
        raise NotImplementedError

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
class KeyBasedAuthentication:
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
        if api_key is not directly provided, secret_key_params must be provided to get the api_key using GetKey method.
        """
        if self.api_key is None:
            self.api_key = GetKey(**self.secret_key_params)
        return self.api_key


@dataclass
class EndpointModels(Model):
    """This class is used to interact with API-based models."""

    url: str = None
    model_name: str = None
    max_tokens: int = 2000
    temperature: float = 0
    top_p: float = 0.95
    num_retries: int = 3
    frequency_penalty: float = 0
    presence_penalty: float = 0
    rate_limit: bool = False
    calls: int = 10
    period: int = 60

    def __post_init__(self):
        if self.rate_limit:
            self.get_response = sleep_and_retry(rate_limited(calls=self.calls, period=self.period)(self.get_response))

    @abstractmethod
    def create_request(self, text_prompt, query_images=None, system_message=None):
        raise NotImplementedError

    @abstractmethod
    def get_response(self, request):
        raise NotImplementedError

    def generate(self, query_text, query_images=None, system_message=None):
        """
        Calls the endpoint to generate the model response.
        args:
            query_text (str): the text prompt to generate the response.
            query_images (list): list of images in base64 bytes format to be included in the request.
            system_message (str): the system message to be included in the request.
        returns:
            response (str): the generated response.
            is_valid (bool): whether the response is valid.
        """
        request = self.create_request(query_text, query_images, system_message)

        attempts = 0
        while attempts < self.num_retries:
            try:
                response = self.get_response(request)
                break
            except Exception as e:
                logging.warning(f"Attempt {attempts+1}/{self.num_retries} failed: {e}")
                response, is_valid, do_return = self.handle_request_error(e)
                if do_return:
                    return response, is_valid
                attempts += 1
        else:
            logging.warning("All attempts failed.")
            return None, False
        return response, True

    @abstractmethod
    def handle_request_error(self, e):
        raise NotImplementedError

@dataclass
class RestEndpointModels(EndpointModels, KeyBasedAuthentication):

    do_sample: bool = True

    def create_request(self, text_prompt, query_images=None, system_message=None):
        data = {
            "input_data": {
                "input_string": [
                    {
                        "role": "user",
                        "content": text_prompt,
                    }
                ],
                "parameters": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "do_sample": self.do_sample,
                    "max_new_tokens": self.max_tokens,
                },
            }
        }
        if system_message:
            data["input_data"]["input_string"] = [{"role": "system", "content": system_message}] + data["input_data"][
                "input_string"
            ]
        if query_images:
            raise NotImplementedError("Images are not supported for GCR endpoints yet.")

        body = str.encode(json.dumps(data))
        # The azureml-model-deployment header will force the request to go to a specific deployment.
        # Remove this header to have the request observe the endpoint traffic rules
        headers = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self.api_key),
            "azureml-model-deployment": self.model_name,
        }

        return urllib.request.Request(self.url, body, headers)

    def get_response(self, request):
        response = urllib.request.urlopen(request)
        res = json.loads(response.read())
        return res["output"]

    def handle_request_error(self, e):
        if isinstance(e, urllib.error.HTTPError):
            logging.info("The request failed with status code: " + str(e.code))
            # Print the headers - they include the requert ID and the timestamp, which are useful for debugging.
            logging.info(e.info())
            logging.info(e.read().decode("utf8", "ignore"))
        return None, False, False


@dataclass
class RestEndpointO1PreviewModelsAzure(EndpointModels):

    do_sample: bool = True

    def __post_init__(self):
        self.bearer_token_provider = get_bearer_token_provider(ManagedIdentityCredential(client_id="205cb331-87f7-4e09-a6dd-70715dec87ec"), "https://cognitiveservices.azure.com/.default")
        super().__post_init__()

    def create_request(self, text_prompt, query_images, system_message):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": text_prompt,
                }
            ],
        }

        body = str.encode(json.dumps(data))
        headers = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self.bearer_token_provider()),
        }

        return urllib.request.Request(self.url, body, headers)

    def get_response(self, request):
        response = urllib.request.urlopen(request)
        res = json.loads(response.read())
        return res["choices"][0]["message"]["content"]

    def handle_request_error(self, e):
        if isinstance(e, urllib.error.HTTPError):
            logging.info("The request failed with status code: " + str(e.code))
            # Print the headers - they include the requert ID and the timestamp, which are useful for debugging.
            logging.info(e.info())
            logging.info(e.read().decode("utf8", "ignore"))
        return None, False, False


@dataclass
class ServerlessAzureRestEndpointModels(EndpointModels, KeyBasedAuthentication):
    """This class can be used for serverless Azure model deployments."""

    """https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-serverless?tabs=azure-ai-studio"""

    stream: str = "false"

    def __post_init__(self):
        super().__post_init__()
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self.api_key),
        }

    @abstractmethod
    def create_request(self, text_prompt, query_images=None, system_message=None):
        # Exact model parameters are model-specific.
        # The method cannot be implemented unless the model being deployed is known.
        raise NotImplementedError

    def get_response(self, request):
        response = urllib.request.urlopen(request)
        res = json.loads(response.read())
        return res["choices"][0]["message"]["content"]

    def handle_request_error(self, e):
        if isinstance(e, urllib.error.HTTPError):
            logging.info("The request failed with status code: " + str(e.code))
            # Print the headers - they include the request ID and the timestamp, which are useful for debugging.
            logging.info(e.info())
            logging.info(e.read().decode("utf8", "ignore"))
        return None, False, False


@dataclass
class LlamaServerlessAzureRestEndpointModels(ServerlessAzureRestEndpointModels, KeyBasedAuthentication):
    """Tested for Llama 3.1 405B Instruct deployments."""

    """See https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-llama?tabs=llama-three for the api reference."""

    use_beam_search: str = "false"
    best_of: int = 1
    skip_special_tokens: str = "false"
    ignore_eos: str = "false"

    def create_request(self, text_prompt, *args):
        data = {
            "messages": [{"role": "user", "content": text_prompt}],
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
class MistralServerlessAzureRestEndpointModels(ServerlessAzureRestEndpointModels, KeyBasedAuthentication):
    """Tested for Mistral Large 2 2407 deployments."""

    """See https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-mistral?tabs=mistral-large#mistral-chat-api for the api reference."""

    safe_prompt: str = "false"

    def __post_init__(self):
        if self.temperature == 0 and self.top_p != 1:
            warning_message = "Top_p must be 1 when using greedy sampling. Temperature zero means greedy sampling. Top_p will be reset to 1. See https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-mistral?tabs=mistral-large#mistral-chat-api for more information."
            logging.warning(warning_message)
            self.top_p = 1
        super().__post_init__()

    def create_request(self, text_prompt, *args):
        data = {
            "messages": [{"role": "user", "content": text_prompt}],
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
class OpenAIModelsMixIn(EndpointModels):
    """
    This class defines the request and response handling for OpenAI models.
    This is an abstract class and should not be used directly. Child classes should implement the get_client
    method and handle_request_error method.
    """

    seed: int = 0
    api_version: str = "2023-06-01-preview"

    @abstractmethod
    def get_client(self):
        raise NotImplementedError

    def create_request(self, prompt, query_images=None, system_message=None):
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        user_content = {"role": "user", "content": prompt}
        if query_images:
            encoded_images = self.base64encode(query_images)
            user_content["content"] = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_images[0]}",
                    },
                },
            ]
        messages.append(user_content)
        return {"messages": messages}

    def get_response(self, request):
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
        openai_response = completion.model_dump()
        return openai_response["choices"][0]["message"]["content"]

    @abstractmethod
    def handle_request_error(self, e):
        raise NotImplementedError


@dataclass
class OpenAIModelsAzure(OpenAIModelsMixIn):
    """This class is used to interact with Azure OpenAI models."""

    def __post_init__(self):
        self.client = self.get_client()

    def get_client(self):
        from openai import AzureOpenAI

        token_provider = get_bearer_token_provider(ManagedIdentityCredential(client_id="205cb331-87f7-4e09-a6dd-70715dec87ec"), "https://cognitiveservices.azure.com/.default")
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
        return None, False, False


@dataclass
class OpenAIModelsO1Azure(OpenAIModelsMixIn):
    """This class is used to interact with Azure OpenAI models."""
    rate_limit: bool = False
    calls: int = 500
    period: int = 60

    def __post_init__(self):
        self.client = self.get_client()
        if self.rate_limit:
            self.get_response = sleep_and_retry(rate_limited(calls=self.calls, period=self.period)(self.get_response))

    def get_client(self):
        from openai import AzureOpenAI

        token_provider = get_bearer_token_provider(ManagedIdentityCredential(client_id="205cb331-87f7-4e09-a6dd-70715dec87ec"), "https://cognitiveservices.azure.com/.default")
        return AzureOpenAI(
            azure_endpoint=self.url,
            api_version=self.api_version,
            azure_ad_token_provider=token_provider,
        )

    def create_request(self, prompt, query_images, system_message):
        messages = []
        user_content = {"role": "user", "content": prompt}
        messages.append(user_content)
        return {"messages": messages}

    def get_response(self, request):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            seed=self.seed,
            **request,
        )
        openai_response = completion.model_dump()
        return openai_response["choices"][0]["message"]["content"]

    def handle_request_error(self, e):
        # if the error is due to a content filter, there is no need to retry
        if e.code == "content_filter":
            logging.warning("Content filtered.")
            response = None
            return response, False, True
        return None, False, False


@dataclass
class OpenAIModelsOAI(OpenAIModelsMixIn, KeyBasedAuthentication):
    """This class is used to interact with OpenAI models dirctly (not through Azure)"""
    
    def __post_init__(self):
        super().__post_init__()
        self.client = self.get_client()
    
    def get_client(self):
        from openai import OpenAI

        return OpenAI(
            api_key=self.api_key,
        )

    def handle_request_error(self, e):
        logging.warning(e)
        return None, False, False


@dataclass
class GeminiModels(EndpointModels, KeyBasedAuthentication):
    """This class is used to interact with Gemini models through the python api."""

    timeout: int = 60

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

    def create_request(self, text_prompt, query_images=None, system_message=None):
        import google.generativeai as genai

        self.model = genai.GenerativeModel(self.model_name, system_instruction=system_message)
        if query_images:
            return [text_prompt] + query_images
        else:
            return text_prompt

    def get_response(self, request):
        self.gemini_response = self.model.generate_content(
            request,
            generation_config=self.gen_config,
            request_options={"timeout": self.timeout},
            safety_settings=self.safety_settings,
        )
        return self.gemini_response.parts[0].text

    def handle_request_error(self, e):
        """Handles exceptions originating from making requests to Gemini through the python api.

        args:
            e (_type_): Exception occurred during getting a response.

        returns:
            _type_: response, is_valid, do_return (False if the call should not be attempted again).
        """
        # Handling cases where the model explicitly blocks prompts and provides a reason for it.
        # In these cases, there is no need to make a new attempt as the model will continue to explicitly block the request, do_return = True.
        if e.__class__.__name__ == "ValueError" and self.gemini_response.prompt_feedback.block_reason > 0:
            logging.warning(
                f"Attempt failed due to explicitly blocked input prompt: {e} Block Reason {self.gemini_response.prompt_feedback.block_reason}"
            )
            return None, False, True
        # Handling cases where the model implicitly blocks prompts and does not provide a reason for it but rather an empty content.
        # In these cases, there is no need to make a new attempt as the model will continue to implicitly block the request, do_return = True.
        elif e.__class__.__name__ == "IndexError" and len(self.gemini_response.parts) == 0:
            logging.warning(f"Attempt failed due to implicitly blocked input prompt and empty model output: {e}")
            return None, False, True
        # Any other case will be re attempted again, do_return = False.
        return None, False, False


@dataclass
class HuggingFaceLM(Model):
    """This class is used to run a self-hosted language model via HuggingFace apis."""

    model_name: str
    device: str = "cpu"
    max_tokens: int = 2000
    temperature: float = 0.001
    top_p: float = 0.95
    do_sample: bool = True
    apply_model_template: bool = True

    def __post_init__(self):
        # The device need to be set before get_model() is called
        self.device = self.pick_available_device()
        self.get_model()

    def get_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
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
        output_ids = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
        )
        sequence_length = inputs["input_ids"].shape[1]
        new_output_ids = output_ids[:, sequence_length:]
        answer_text = self.tokenizer.batch_decode(
            new_output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return answer_text

    def generate(self, text_prompt, query_images=None, system_message=None):

        if text_prompt is None:
            return None, False

        if self.apply_model_template:
            text_prompt = self.model_template_fn(text_prompt, system_message)

        try:
            answer_text = self._generate(text_prompt, query_images)

        except Exception as e:
            logging.warning(e)
            return None, False

        return answer_text, True

    def model_template_fn(self, text_prompt, system_message=None):
        return system_message + " " + text_prompt if system_message else text_prompt


@dataclass
class Phi3HF(HuggingFaceLM):
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
class LLaVAHuggingFaceMM(HuggingFaceLM):
    """This class is used to run a self-hosted LLaVA model via HuggingFace apis."""

    quantize: bool = False
    use_flash_attn: bool = False

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

        if self.quantize:
            from transformers import BitsAndBytesConfig

            logging.info("Quantizing model")
            # specify how to quantize the model
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        if "v1.6" in self.model_name:
            if self.quantize:
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    quantization_config=quantization_config,
                    device_map=self.device,
                    use_flash_attention_2=self.use_flash_attn,
                )
            else:
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map=self.device,
                    use_flash_attention_2=self.use_flash_attn,
                )

            self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
        else:
            if self.quantize:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    quantization_config=quantization_config,
                    device_map=self.device,
                    use_flash_attention_2=self.use_flash_attn,
                )
            else:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map=self.device,
                    use_flash_attention_2=self.use_flash_attn,
                )

            self.processor = AutoProcessor.from_pretrained(self.model_name)

    def _generate(self, text_prompt, query_images=None):
        inputs = self.processor(text=text_prompt, images=query_images, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
        )

        sequence_length = inputs["input_ids"].shape[1]
        new_output_ids = output_ids[:, sequence_length:]
        answer_text = self.processor.batch_decode(
            new_output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return answer_text

    def generate(self, text_prompt, query_images=None, system_message=None):

        if len(query_images) > 1:
            logging.error(f"Not implemented for more than 1 image. {len(query_images)} images are in the prompt")
            return None, False

        return super().generate(text_prompt, query_images, system_message)

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
class LLaVA(LLaVAHuggingFaceMM):
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

        answer_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return answer_text


@dataclass
class ClaudeModels(EndpointModels, KeyBasedAuthentication):
    """This class is used to interact with Claude models through the python api."""


    timeout: int = 60

    def __post_init__(self):
        super().__post_init__()
        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def create_request(self, prompt, query_images=None, system_message=None):
        messages = []

        user_content = {"role": "user", "content": prompt}

        if query_images:
            encoded_images = self.base64encode(query_images)
            user_content["content"] = [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": encoded_images[0],
                    },
                },
            ]
        messages.append(user_content)

        if system_message:
            return {"messages": messages, "system": system_message}
        else:
            return {"messages": messages}

    def get_response(self, request):
        completion = self.client.messages.create(
            model=self.model_name,
            **request,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        claude_response = completion.content[0].text
        return claude_response

    def handle_request_error(self, e):
        return None, False, False
