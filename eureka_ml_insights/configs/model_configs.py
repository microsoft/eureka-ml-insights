""" This module contains config objects for the models used in the experiments. To use these configs, make sure to
replace the placeholders with your own keys.json file, secret key names, and endpint URLs where applicable. 
You can also add your custom models here by following the same pattern as the existing configs. """

from eureka_ml_insights.models import (
    ClaudeModel,
    GeminiModel,
    LlamaServerlessAzureRestEndpointModel,
    LLaVAModel,
    LLaVAHuggingFaceModel,
    MistralServerlessAzureRestEndpointModel,
    DirectOpenAIModel,
    AzureOpenAIO1Model,
    DirectOpenAIO1Model,
    RestEndpointModel,
)

from .config import ModelConfig

# OpenAI models
OPENAI_SECRET_KEY_PARAMS = {
    "key_name": "your_openai_secret_key_name",
    "local_keys_path": "keys/keys.json",
    "key_vault_url": None,
}

OAI_O1_PREVIEW_CONFIG = ModelConfig(
    DirectOpenAIO1Model,
    {
        "model_name": "o1-preview",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

OAI_O1_PREVIEW_AUZRE_CONFIG = ModelConfig(
    AzureOpenAIO1Model,
    {
        "model_name": "o1-preview",
        "url": "your/endpoint/url",
        "api_version": "2024-08-01-preview",
    }
)

OAI_GPT4_1106_PREVIEW_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "model_name": "gpt-4-1106-preview",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

OAI_GPT4V_1106_VISION_PREVIEW_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "model_name": "gpt-4-1106-vision-preview",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

OAI_GPT4V_TURBO_2024_04_09_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "model_name": "gpt-4-turbo-2024-04-09",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

OAI_GPT4O_2024_05_13_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "model_name": "gpt-4o-2024-05-13",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)


# Gemini models
GEMINI_SECRET_KEY_PARAMS = {
    "key_name": "your_gemini_secret_key_name",
    "local_keys_path": "keys/keys.json",
    "key_vault_url": None,
}

GEMINI_V15_PRO_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-1.5-pro",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
    },
)

GEMINI_V1_PRO_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-1.0-pro",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
    },
)

# Claude models
CLAUDE_SECRET_KEY_PARAMS = {
    "key_name": "your_claude_secret_key_name",
    "local_keys_path": "keys/keys.json",
    "key_vault_url": None,
}

CLAUDE_3_OPUS_CONFIG = ModelConfig(
    ClaudeModel,
    {
        "model_name": "claude-3-opus-20240229",
        "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
    },
)

CLAUDE_3_5_SONNET_CONFIG = ModelConfig(
    ClaudeModel,
    {
        "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
        "model_name": "claude-3-5-sonnet-20240620",
    },
)

# LLAVA models
LLAVAHF_V16_34B_CONFIG = ModelConfig(
    LLaVAHuggingFaceModel,
    {"model_name": "llava-hf/llava-v1.6-34b-hf", "use_flash_attn": True},
)

LLAVAHF_V15_7B_CONFIG = ModelConfig(
    LLaVAHuggingFaceModel,
    {"model_name": "llava-hf/llava-1.5-7b-hf", "use_flash_attn": True},
)

LLAVA_V16_34B_CONFIG = ModelConfig(
    LLaVAModel,
    {"model_name": "liuhaotian/llava-v1.6-34b", "use_flash_attn": True},
)

LLAVA_V15_7B_CONFIG = ModelConfig(
    LLaVAModel,
    {"model_name": "liuhaotian/llava-v1.5-7b", "use_flash_attn": True},
)

# Llama models

LLAMA3_1_70B_INSTRUCT_CONFIG = ModelConfig(
    RestEndpointModel,
    {
        "url": "your/endpoint/url",
        "secret_key_params": {
            "key_name": "your_llama_secret_key_name",
            "local_keys_path": "keys/keys.json",
            "key_vault_url": None,
        },
        "model_name": "meta-llama-3-1-70b-instruct",
    },
)

LLAMA3_1_405B_INSTRUCT_CONFIG = ModelConfig(
    LlamaServerlessAzureRestEndpointModel,
    {
        "url": "your/endpoint/url",
        "secret_key_params": {
            "key_name": "your_llama_secret_key_name",
            "local_keys_path": "keys/keys.json",
            "key_vault_url": None,
        },
        "model_name": "Meta-Llama-3-1-405B-Instruct",
    },
)

# Mistral Endpoints
AIF_NT_MISTRAL_LARGE_2_2407_CONFIG = ModelConfig(
    MistralServerlessAzureRestEndpointModel,
    {
        "url": "your/endpoint/url",
        "secret_key_params": {
            "key_name": "your_mistral_secret_key_name",
            "local_keys_path": "keys/keys.json",
            "key_vault_url": None,
        },
        "model_name": "Mistral-large-2407",
    },
)
