"""This module provides classes to run self-hosted models via HuggingFace APIs, including specialized templates for various model families such as Phi-3, Phi-4, and LLaVA."""

from dataclasses import dataclass
import logging
import time

try:
    from ..model import Model
except ImportError:
    class Model:
        pass


@dataclass
class HuggingFaceModel(Model):
    """Runs a self-hosted language model via HuggingFace APIs.

    This class handles loading and running a HuggingFace language model locally
    with optional quantization and flash attention usage.

    Attributes:
        model_name (str): The name of the HuggingFace model to use.
        device (str): The device to use for inference (e.g., "cpu" or "cuda").
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature for sampling-based generation.
        top_p (float): The top-p (nucleus) sampling parameter.
        do_sample (bool): Whether to sample or not, setting to False uses greedy decoding.
        apply_model_template (bool): If True, applies a template to the prompt before generating.
        quantize (bool): Whether to quantize the model for memory savings.
        use_flash_attn (bool): Whether to use flash attention 2 if supported.
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
        """Initializes model-related attributes after the dataclass has been populated.

        This method sets the device by picking an available GPU or falling back
        to CPU before actually loading the model.
        """
        # The device needs to be set before get_model() is called
        self.device = self.pick_available_device()
        self.get_model()

    def get_model(self):
        """Loads the HuggingFace model and tokenizer.

        If quantization is enabled, applies 4-bit quantization. Otherwise, loads
        the model with standard precision. Tokenizer is also loaded using the
        same model name.
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
        """Selects the device with the lowest GPU utilization or defaults to CPU.

        Enumerates all available GPU devices, checks utilization, and returns the
        device with the lowest utilization. If no GPU is available, returns 'cpu'.

        Returns:
            str: Name of the chosen device (e.g., 'cuda:0' or 'cpu').
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
        """Generates text given a prompt and optional images.

        Args:
            text_prompt (str): The text prompt for the model to process.
            query_images (list, optional): A list of images (if supported by the model).

        Returns:
            dict: A dictionary containing:
                "model_output" (str): The generated text response.
                "response_time" (float): The time taken for the generation.
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
        """Generates a text response, optionally applying a model-specific template.

        Args:
            text_prompt (str): The text prompt for generation.
            query_images (list, optional): A list of images (if supported by the model).
            system_message (str, optional): A system message to be prepended or otherwise
                integrated into the template.

        Returns:
            dict: A dictionary containing:
                "model_output" (str): The generated text response.
                "response_time" (float): The time taken for generation.
                "is_valid" (bool): Whether the generation was successful.
                "n_output_tokens" (int): Number of tokens in the generated output.
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
        """Applies a basic template to the prompt.

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
    """Runs a self-hosted PHI3 model via HuggingFace APIs.

    Extends HuggingFaceModel, applying a PHI3-specific prompt template.
    """

    def __post_init__(self):
        """Initializes and checks if the model name is a PHI3 model."""
        super().__post_init__()
        if "microsoft/Phi-3" not in self.model_name:
            logging.warning(
                "This model class applies a template to the prompt that is specific to Phi-3 models"
                " but your model is not a Phi-3 model."
            )

    def model_template_fn(self, text_prompt, system_message=None):
        """Applies the PHI3-specific template to the prompt.

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
    """Runs a self-hosted PHI4 model via HuggingFace APIs.

    Extends HuggingFaceModel, applying a PHI4-specific prompt template.
    """

    def __post_init__(self):
        """Initializes and checks if the model name is a PHI4 model."""
        super().__post_init__()
        if "microsoft/phi-4" not in self.model_name:
            logging.warning(
                "This model class applies a template to the prompt that is specific to Phi-4 models"
                " but your model is not a Phi-4 model."
            )

    def model_template_fn(self, text_prompt, system_message=None):
        """Applies the PHI4-specific template to the prompt.

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
    """Runs a self-hosted LLaVA model via HuggingFace APIs.

    Extends HuggingFaceModel, applying an image-based prompt template for LLaVA.
    """

    def __post_init__(self):
        """Initializes and checks if the model name is a LLaVA model."""
        super().__post_init__()
        if "llava" not in self.model_name:
            logging.warning(
                "This model class applies a template to the prompt that is specific to LLAVA models"
                " but your model is not a LLAVA model."
            )

    def get_model(self):
        """Loads the LLaVA model and processor.

        If quantization is enabled, applies 4-bit quantization. Otherwise, loads
        the model with standard precision. The appropriate LLaVA variant is chosen
        depending on the model name (v1.6, etc.).
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
        """Generates a response given a text prompt and optional image(s).

        Args:
            text_prompt (str): The text prompt to generate a response for.
            query_images (list, optional): A list with an image to use as context.

        Returns:
            dict: A dictionary containing:
                "model_output" (str): The generated text response.
                "response_time" (float): The time taken for the generation.
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
        """Generates a response using a LLaVA model, with optional images.

        Args:
            text_prompt (str): The text prompt for generation.
            query_images (list, optional): A list containing a single image to use.
            system_message (str, optional): Additional system message or instruction.

        Returns:
            dict: A dictionary containing generation results. If multiple images are
                provided, returns an error message in the dictionary.
        """
        if query_images and len(query_images) > 1:
            logging.error(f"Not implemented for more than 1 image. {len(query_images)} images are in the prompt")
            return {"model_output": None, "is_valid": False, "response_time": None, "n_output_tokens": None}

        return super().generate(text_prompt, query_images=query_images, system_message=system_message)

    def model_template_fn(self, text_prompt, system_message=None):
        """Applies an image-based template to the text prompt for LLaVA models.

        The exact template depends on the LLaVA model variant (v1.6, v1.6-mistral, etc.).

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
    """Runs a self-hosted LLaVA model using the LLaVA package.

    Extends LLaVAHuggingFaceModel to handle model loading and inference through
    the dedicated LLaVA package utilities.

    Attributes:
        model_base (str): The base model to use for LLaVA.
        num_beams (int): The number of beams for beam search decoding.
    """

    model_base: str = None
    num_beams: int = 1

    def __post_init__(self):
        """Initializes the LLaVA model after the dataclass has been populated."""
        super().__post_init__()

    def get_model(self):
        """Loads the LLaVA model and tokenizer using the LLaVA package.

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
        """Generates a response using the LLaVA package.

        Args:
            text_prompt (str): The text prompt to process.
            query_images (list, optional): A list of images for multimodal generation.
            system_message (str, optional): Additional system message for context.

        Returns:
            dict: A dictionary containing:
                "model_output" (str): The generated text response.
                "response_time" (float): The time taken for generation.
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