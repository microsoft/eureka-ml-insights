""" This module contains config objects for the models used in the experiments. To use these configs, make sure to
replace the placeholders with your own keys.json file, secret key names, and endpint URLs where applicable. 
You can also add your custom models here by following the same pattern as the existing configs. """

from eureka_ml_insights.models import (
    AzureOpenAIOModel,
    ClaudeModel,
    DirectOpenAIModel,
    DirectOpenAIOModel,
    GeminiModel,
    LlamaServerlessAzureRestEndpointModel,
    LLaVAHuggingFaceModel,
    LLaVAModel,
    Phi4HFModel,
    MistralServerlessAzureRestEndpointModel,
    RestEndpointModel,
    TogetherModel,
    TestModel,
)
from eureka_ml_insights.models.models import AzureOpenAIModel

from .config import ModelConfig

# For models that require secret keys, you can store the keys in a json file and provide the path to the file
# in the secret_key_params dictionary. OR you can provide the key name and key vault URL to fetch the key from Azure Key Vault.
# You don't need to provide both the key_vault_url and local_keys_path. You can provide one of them based on your setup.


# Test model
TEST_MODEL_CONFIG = ModelConfig(TestModel, {})

# Together models
TOGETHER_SECRET_KEY_PARAMS = {
    "key_name": "your_togetherai_secret_key_name",
    "local_keys_path": "keys/keys.json",
    "key_vault_url": None,
}

TOGETHER_DEEPSEEK_R1_CONFIG = ModelConfig(
    TogetherModel,
    {
        "model_name": "deepseek-ai/DeepSeek-R1",
        "secret_key_params": TOGETHER_SECRET_KEY_PARAMS,
        "temperature": 1.0,
        # high max token limit for deep seek
        # otherwise the answers may be cut in the middle
        "max_tokens": 65536
    },
)
# OpenAI models

OPENAI_SECRET_KEY_PARAMS = {
    "key_name": "openai",
    "local_keys_path": "keys/aifeval-vault-azure-net.json",
    "key_vault_url": None,
}

OAI_O3_MINI_HIGH_CONFIG = ModelConfig(
    DirectOpenAIOModel,
    {
        "model_name": "o3-mini-2025-01-31",
        "reasoning_effort": "high", 
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

OAI_O3_MINI_CONFIG = ModelConfig(
    DirectOpenAIOModel,
    {
        "model_name": "o3-mini-2025-01-31",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

OAI_O1_CONFIG = ModelConfig(
    DirectOpenAIOModel,
    {
        "model_name": "o1",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

OAI_O1_PREVIEW_CONFIG = ModelConfig(
    DirectOpenAIOModel,
    {
        "model_name": "o1-preview",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

OAI_O1_PREVIEW_AUZRE_CONFIG = ModelConfig(
    AzureOpenAIOModel,
    {
        "model_name": "o1-preview",
        "url": "your/endpoint/url",
        "api_version": "2024-08-01-preview",
    },
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

OAI_GPT4O_2024_11_20_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "model_name": "gpt-4o-2024-11-20",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

OAI_GPT4O_MINI_2024_07_18_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "model_name": "gpt-4o-mini-2024-07-18",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

############### TRAPI models ####################

TRAPI_AIF_O3_MINI_CONFIG = ModelConfig(
    AzureOpenAIOModel,
    {
        "url": "https://trapi.research.microsoft.com/msraif/shared",
        # o1 models only work with > 2024-12-01-preview api version
        "api_version": '2025-01-01-preview',
        "model_name": "o3-mini_2025-01-31",
        "reasoning_effort": "high",
        "auth_scope": "api://trapi/.default"
    },
)


# #################################################


# # Gemini models
# GEMINI_SECRET_KEY_PARAMS = {
#     "key_name": "your_gemini_secret_key_name",
#     "local_keys_path": "keys/keys.json",
#     "key_vault_url": None,
# }

# GEMINI_V2_FLASH_THINKING_EXP_0121_CONFIG = ModelConfig(
#     GeminiModel,
#     {
#         "model_name": "gemini-2.0-flash-thinking-exp-01-21",
#         "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
# 	    "max_tokens": 32768
#     },
# )

# GEMINI_V2_PRO_EXP_0205_CONFIG = ModelConfig(
#     GeminiModel,
#     {
#         "model_name": "gemini-2.0-pro-exp-02-05",
#         "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
#     },
# )

# GEMINI_V15_PRO_CONFIG = ModelConfig(
#     GeminiModel,
#     {
#         "model_name": "gemini-1.5-pro",
#         "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
#     },
# )


####################

# Gemini models

GEMINI_SECRET_KEY_PARAMS = {
    "key_name": "aif-eval-gemini",
    # currently we have three keys: "aif-eval-gemini-firstproject", "aif-eval-gemini", "aif-eval-gemini-aifevalunderstandproject"
    # rotate between these if you get '429 Resource has been exhausted (e.g. check quota)' 
    "local_keys_path": "keys/aifeval-vault-azure-net.json",
    "key_vault_url": "https://aifeval.vault.azure.net",
}

GEMINI_V2_PRO_T1_M4096_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-2.0-pro-exp-02-05",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
        "temperature":1.0,
        "max_tokens":4096,

    },
)


GEMINI_V2_FLASH_THINKING_EXP_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-2.0-flash-thinking-exp-1219",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
        "temperature": 1.0,
        "max_tokens": 32768        
    },
)

GEMINI_V2_FLASH_THINKING_EXP_0121_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-2.0-flash-thinking-exp-01-21",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
        "temperature": 1.0,
        "max_tokens": 32768
    },
)

GEMINI_EXP_1206_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-exp-1206",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
        "temperature": 1.0,
        "max_tokens": 4096
    },
)


# # Gemini models
# GEMINI_SECRET_KEY_PARAMS = {
#     "key_name": "your_gemini_secret_key_name",
#     "local_keys_path": "keys/keys.json",
#     "key_vault_url": None,
# }

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

CLAUDE_3_5_SONNET_20241022_CONFIG = ModelConfig(
    ClaudeModel,
    {
        "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
        "model_name": "claude-3-5-sonnet-20241022",
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

# Phi Models
PHI4_HF_CONFIG = ModelConfig(
    Phi4HFModel,
    {
        "model_name": "microsoft/phi-4",
        "use_flash_attn": True,
    },
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
        "timeout": 600,
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
        "timeout": 600,
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
