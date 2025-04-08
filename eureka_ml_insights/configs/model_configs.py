""" This module contains config objects for the models used in the experiments. To use these configs, make sure to
replace the placeholders with your own keys.json file, secret key names, and endpint URLs where applicable. 
You can also add your custom models here by following the same pattern as the existing configs. """

from eureka_ml_insights.models import (
    #AzureOpenAIO1Model,
    AzureOpenAIModel,
    ClaudeModel,
    DirectOpenAIModel,
    #DirectOpenAIO1Model,
    DirectOpenAIOModel,
    GeminiModel,
    LlamaServerlessAzureRestEndpointModel,
    LLaVAHuggingFaceModel,
    LLaVAModel,
    MistralServerlessAzureRestEndpointModel,
    RestEndpointModel,
    DeepseekR1ServerlessAzureRestEndpointModel,
    #TnRModels,
    ClaudeReasoningModel,
    TogetherModel,
)

from eureka_ml_insights.models import AzureOpenAIOModel as AzureOpenAIO1Model
from eureka_ml_insights.models import DirectOpenAIOModel as DirectOpenAIO1Model
from eureka_ml_insights.models import AzureOpenAIOModel 
from .config import ModelConfig

# For models that require secret keys, you can store the keys in a json file and provide the path to the file
# in the secret_key_params dictionary. OR you can provide the key name and key vault URL to fetch the key from Azure Key Vault.
# You don't need to provide both the key_vault_url and local_keys_path. You can provide one of them based on your setup.

# OpenAI models

'''
OPENAI_SECRET_KEY_PARAMS = {
    "key_name": "your_openai_secret_key_name",
    "local_keys_path": "keys/keys.json",
    "key_vault_url": None,
}
'''

OPENAI_SECRET_KEY_PARAMS = {
    "key_name": "openai",
    "local_keys_path": "keys/aifeval-vault-azure-net.json",
    "key_vault_url": "https://aifeval.vault.azure.net",
}


OAI_O1_PREVIEW_CONFIG = ModelConfig(
    DirectOpenAIO1Model,
    {
        "model_name": "o1-preview",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

OAI_O1_MINI_CONFIG = ModelConfig(
    DirectOpenAIO1Model,
    {
        "model_name": "o1-mini-2024-09-12",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

OAI_O1_PREVIEW_AUZRE_CONFIG = ModelConfig(
    AzureOpenAIO1Model,
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

OAI_GPT4O_2024_11_20_CONFIG  = ModelConfig(
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

# Azure OAI models
## Azure OAI models -- TNR Models

TNR_SECRET_KEY_PARAMS = {
    "key_name": "tnrllmproxy",
    "local_keys_path": "keys/aifeval-vault-azure-net.json",
    "key_vault_url": "https://aifeval.vault.azure.net",
}

GCRAOAI8SW1_AZURE_OAI_O1_PREVIEW_CONFIG = ModelConfig(
    AzureOpenAIO1Model,
    {
        "url": "https://gcraoai8sw1.openai.azure.com/",
        "model_name": "o1-preview",
        "api_version": "2024-08-01-preview",
    }
)

GCRAOAI8SW1_AZURE_OAI_O1_MINI_CONFIG = ModelConfig(
    AzureOpenAIO1Model,
    {
        "url": "https://gcraoai8sw1.openai.azure.com/",
        "model_name": "o1-mini",
        "api_version": "2024-08-01-preview",
    }
)

GCRAOAI8SW1_AZURE_OAI_GPT4O_CONFIG = ModelConfig(
    AzureOpenAIO1Model,
    {
        "url": "https://gcraoai8sw1.openai.azure.com/",
        "model_name": "gpt-4o",
        "api_version": "2024-08-01-preview",
        "temperature": 1.0,

    }
)


GCRAOAI8SW1_AZURE_OAI_GPT4_T1_CONFIG = ModelConfig(
    AzureOpenAIO1Model,
    {
        "url": "https://gcraoai8sw1.openai.azure.com/",
        "model_name": "gpt-4",
        "api_version": "2024-08-01-preview",
        "temperature": 1.0,

    }
)

AzureOpenAIModel

"""
TNR_GPT4_1106_PREVIEW_CONFIG = ModelConfig(
    TnRModels,
    {
        "url": "https://trapi.research.microsoft.com/gcr/shared/nj/",
        "secret_key_params": TNR_SECRET_KEY_PARAMS,
        "model_name": "gpt-4",
    },
)

TNR_GPT4_VISION_PREVIEW_CONFIG = ModelConfig(
    TnRModels,
    {
        "url": "https://trapi.research.microsoft.com/gcr/shared/nj/",
        "secret_key_params": TNR_SECRET_KEY_PARAMS,
        "model_name": "gpt-4-turbo-v",
    },
)

TNR_GPT4V_TURBO_2024_04_09_CONFIG = ModelConfig(
    TnRModels,
    {
        "url": "https://trapi.research.microsoft.com/gcr/shared/nj/",
        "secret_key_params": TNR_SECRET_KEY_PARAMS,
        "model_name": "gpt-4-turbo",
    },
)

TNR_GPT4O_2024_05_13_CONFIG = ModelConfig(
    TnRModels,
    {
        "url": "https://trapi.research.microsoft.com/gcr/shared/nj/",
        "secret_key_params": TNR_SECRET_KEY_PARAMS,
        "model_name": "gpt-4o",
    },
)
"""

# Gemini models
'''
GEMINI_SECRET_KEY_PARAMS = {
    "key_name": "your_gemini_secret_key_name",
    "local_keys_path": "keys/keys.json",
    "key_vault_url": None,
}
'''

#     "key_name": "aif-eval-gemini-aime",


GEMINI_SECRET_KEY_PARAMS = {
    "key_name": "aif-eval-gemini-vl",
    "local_keys_path": "keys/aifeval-vault-azure-net.json",
    "key_vault_url": "https://aifeval.vault.azure.net",
}

GEMINI_V15_PRO_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-1.5-pro",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
    },
)


GEMINI_V15_PRO_T1_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-1.5-pro",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
        "temperature":1.0,
    },
)

GEMINI_EXP_1206_T1_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-exp-1206",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
        "temperature":1.0,
    },
)


GEMINI_EXP_1121_T1_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-exp-1121",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
        "temperature":1.0,
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
'''
CLAUDE_SECRET_KEY_PARAMS = {
    "key_name": "your_claude_secret_key_name",
    "local_keys_path": "keys/keys.json",
    "key_vault_url": None,
}
'''

CLAUDE_SECRET_KEY_PARAMS = {
    #"key_name": "claude-besmira-gmail-account",
    "key_name":"claude-aifeval-account-2",
    "local_keys_path": "keys/aifeval-vault-azure-net.json",
    "key_vault_url": "https://aifeval.vault.azure.net",
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

CLAUDE_3_5_SONNET_T1_CONFIG = ModelConfig(
    ClaudeModel,
    {
        "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
        "model_name": "claude-3-5-sonnet-20240620",
        "temperature":1.0,
    },
)

CLAUDE_3_5_SONNET_SEARCH_T1_CONFIG = ModelConfig(
    ClaudeModel,
    {
        "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
        "model_name": "claude-3-5-sonnet-20241022",
        "temperature": 1.0,
    },
)

CLAUDE_3_5_SONNET_SEARCH_CONFIG = ModelConfig(
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



AIF_NT_MISTRAL_LARGE_2_2407_T1_CONFIG = ModelConfig(
    MistralServerlessAzureRestEndpointModel,
    {
        "url": "https://Mistral-large-2407-aifeval.eastus.models.ai.azure.com/v1/chat/completions",
        "secret_key_params": {
            "key_name": "aif-nt-mistral-large-2-2407",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
        "model_name": "Mistral-large-2407-aifeval",
                "temperature": 1.0,

    },
)


GCR_LLAMA3_1_70B_INSTRUCT_CONFIG = ModelConfig(
    RestEndpointModel,
    {
        "url": "https://gcr-llama31-70b-instruct.westus3.inference.ml.azure.com/score",
        "secret_key_params": {
            "key_name": "meta-llama-3-1-70b-instruct-1",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
        "model_name": "meta-llama-3-1-70b-instruct-1",
                        "temperature": 1.0,

    },
)

AIF_NT_LLAMA3_1_405B_INSTRUCT_Token4K_CONFIG = ModelConfig(
    LlamaServerlessAzureRestEndpointModel,
    {
        "url": "https://Meta-Llama-3-1-405B-Instruct-aif.eastus.models.ai.azure.com/v1/chat/completions",
        "secret_key_params": {
            "key_name": "aif-nt-meta-llama-3-1-405b-instruct-1",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
        "model_name": "Meta-Llama-3-1-405B-Instruct-aif",
        "temperature": 1.0,
        "max_tokens":4096,

    },
)


AIF_NT_LLAMA3_1_405B_INSTRUCT_T0_M4096_CONFIG = ModelConfig(
    LlamaServerlessAzureRestEndpointModel,
    {
        "url": "https://Meta-Llama-3-1-405B-Instruct-aif.eastus.models.ai.azure.com/v1/chat/completions",
        "secret_key_params": {
            "key_name": "aif-nt-meta-llama-3-1-405b-instruct-1",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
        "model_name": "Meta-Llama-3-1-405B-Instruct-aif",
        "temperature": 0.0,
        "max_tokens":4096,

    },
)


AIF_NT_LLAMA3_1_405B_INSTRUCT_WESTUS3_T1_M4096_CONFIG = ModelConfig(
    LlamaServerlessAzureRestEndpointModel,
    {
        "url": "https://Meta-Llama-3-1-405B-Instruct-2.westus3.models.ai.azure.com/v1/chat/completions",
        "secret_key_params": {
            "key_name": "aif-nt-meta-llama-3-1-405b-instruct-westus3",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
        "temperature": 1.0,
        "max_tokens":4096,
    },
)


GEMINI_V15_PRO_T1_M4096_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-1.5-pro",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
        "temperature":1.0,
        "max_tokens":4096,

    },
)

GEMINI_V15_PRO_T0_M4096_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-1.5-pro",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
        "temperature":0.0,
        "max_tokens":4096,

    },
)



GEMINI_V2_FLASH_THINKING_EXP_T0_M4096_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-2.0-flash-thinking-exp-1219",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
        "temperature":0.0,
        "max_tokens":4096,
    },
)


GEMINI_EXP_1206_T1_M4096_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-exp-1206",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
        "temperature":1.0,
        "max_tokens":4096,

    },
)

GEMINI_EXP_1206_T0_M4096_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-exp-1206",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
        "temperature":0.0,
        "max_tokens":4096,

    },
)



GEMINI_V2_FLASH_T1_M4096_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-2.0-flash-exp",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
        "temperature": 1.0,
        "max_tokens":4096,
    },
)

GEMINI_V2_FLASH_T0_M4096_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-2.0-flash-exp",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
        "temperature": 0.0,
        "max_tokens":4096,
    },
)



OAI_O1_CONFIG = ModelConfig(
    DirectOpenAIO1Model,
    {
        "model_name": "o1",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)



OAI_GPT4O_2024_11_20_T1_M4096_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "model_name": "gpt-4o-2024-11-20",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
        "temperature":1.0,
        "max_tokens":4096,
    },
)

OAI_GPT4O_2024_11_20_T0_M4096_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "model_name": "gpt-4o-2024-11-20",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
        "temperature":0.0,
        "max_tokens":4096,
    },
)

CLAUDE_3_7_SONNET_0219_T1_M4096_CONFIG = ModelConfig(
    ClaudeModel,
    {
        "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
        "model_name": "claude-3-7-sonnet-20250219",
        "temperature": 1.0,
        "max_tokens":4096,
    },
)


CLAUDE_3_5_SONNET_1022_T05_M4096_CONFIG = ModelConfig(
    ClaudeModel,
    {
        "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
        "model_name": "claude-3-5-sonnet-20241022",
        "temperature": 0.5,
        "max_tokens":4096,
    },
)

CLAUDE_3_5_SONNET_1022_T0_M4096_CONFIG = ModelConfig(
    ClaudeModel,
    {
        "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
        "model_name": "claude-3-5-sonnet-20241022",
        "temperature": 0.0,
        "max_tokens":4096,
    },
)


TRAPI_O1_CONFIG = ModelConfig(
    AzureOpenAIO1Model,
    {
        "url": "https://trapi.research.microsoft.com/msraif/shared",
        # o1 models only work with 2024-12-01-preview api version
        "api_version": '2024-12-01-preview',
        "model_name": "o1_2024-12-17",
        "auth_scope": "api://trapi/.default"
    },
)



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
        "model_name": "gemini-2.0-flash-thinking-exp-01-21",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
        "temperature": 1.0,
        "max_tokens": 32768
    },
)

'''
MSR_LIT_O3_mini_reasoning_1_CONFIG = ModelConfig(
    AzureOpenAIOModel,
    {
        "url": "https://reasoning-eastus2.openai.azure.com/",
        "api_version": '2024-12-01-preview',
        ## o3-mini: o3-mini-reasoning-1, o3-mini-reasoning-2
        "model_name": "o3-mini-reasoning-1",
        "auth_scope": "https://cognitiveservices.azure.com/.default",
    },
)
'''

OAI_GPT4O_2024_08_06_T1_M4096_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "model_name": "gpt-4o-2024-08-06",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
        "temperature": 1.0,
        "max_tokens": 4096
    },
)

OAI_O3_MINI_HIGH_CONFIG = ModelConfig(
    DirectOpenAIOModel,
    {
        "model_name": "o3-mini-2025-01-31",
        "reasoning_effort": "high",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

TOGETHER_SECRET_KEY_PARAMS = {
    # we do not have a key yet for together models
    "key_name": "togetherai",
    "local_keys_path": "keys/aifeval-vault-azure-net.json",
    "key_vault_url": None,
}


TOGETHER_DEEPSEEK_R1_CONFIG = ModelConfig(
    TogetherModel,
    {
        "model_name": "deepseek-ai/DeepSeek-R1",
        "secret_key_params": TOGETHER_SECRET_KEY_PARAMS,
        "temperature": 0.6,
        # high max token limit for deep seek
        # otherwise the answers may be cut in the middle
        "max_tokens": 65536
    },
)


MSR_LIT_DEEPSEEK_R1_CONFIG = ModelConfig(
    DeepseekR1ServerlessAzureRestEndpointModel,
    {
        "url": "https://deepseek-r1-reasoning.westus.models.ai.azure.com/v1/chat/completions",
        "secret_key_params": {
            "key_name": "lit-deepseek-r1",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
        "max_tokens": 32768,
        "timeout": 600
    },
)




# Cleaned models

CLAUDE_3_5_SONNET_1022_T1_M4096_CONFIG = ModelConfig(
    ClaudeModel,
    {
        "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
        "model_name": "claude-3-5-sonnet-20241022",
        "temperature": 1.0,
        "max_tokens":4096,
    },
)


CLAUDE_3_7_SONNET_THINKING_CONFIG = ModelConfig(
    ClaudeReasoningModel,
    {
        "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
        "model_name": "claude-3-7-sonnet-20250219",
        "thinking_enabled": True,
        "thinking_budget": 60000,
        "max_tokens": 64000, # This number should always be higher than the thinking budget
        "temperature": 1.0, # As of 03/08/2025, thinking only works with temperature 1.0
        "timeout": 600, # We set a timeout of 10 minutes for thinking
    },
)

AIF_NT_LLAMA3_1_405B_INSTRUCT_WESTUS3_T1_M4096_CONFIG = ModelConfig(
    LlamaServerlessAzureRestEndpointModel,
    {
        "url": "https://Meta-Llama-3-1-405B-Instruct-2.westus3.models.ai.azure.com/v1/chat/completions",
        "secret_key_params": {
            "key_name": "aif-nt-meta-llama-3-1-405b-instruct-westus3",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
        "temperature": 1.0,
        "max_tokens":4096,
    },
)



MSR_LIT_O3_mini_reasoning_1_CONFIG = ModelConfig(
    AzureOpenAIOModel,
    {
        "url": "https://reasoning-eastus2.openai.azure.com/",
        "api_version": '2024-12-01-preview',
        ## o3-mini: o3-mini-reasoning-1, o3-mini-reasoning-2
        "model_name": "o3-mini-reasoning-1",
        "auth_scope": "https://cognitiveservices.azure.com/.default",
        "reasoning_effort": "high"
    },
)



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
        "model_name": "gemini-2.0-flash-thinking-exp-01-21",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
        "temperature": 1.0,
        "max_tokens": 32768
    },
)


OAI_GPT4O_2024_08_06_T1_M4096_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "model_name": "gpt-4o-2024-08-06",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
        "temperature":1.0,
        "max_tokens":4096,
    },
)


OAI_O1_CONFIG = ModelConfig(
    DirectOpenAIOModel,
    {
        "model_name": "o1-2024-12-17",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)


MSR_LIT_O1_reasoning_1_CONFIG = ModelConfig(
    AzureOpenAIOModel,
    {
        "url": "https://reasoning-eastus2.openai.azure.com/",
        "api_version": '2024-12-01-preview',
        ## o1: o1-reasoning-1, o1-reasoning-2
        "model_name": "o1-reasoning-3",
        "auth_scope": "https://cognitiveservices.azure.com/.default",        
    },
)


TRAPI_GPT4O_2024_08_06_CONFIG = ModelConfig(
    AzureOpenAIModel,
    {
        "url": "https://trapi.research.microsoft.com/msraif/shared",
        "api_version": '2024-10-21',
        "model_name": "gpt-4o_2024-08-06",
        "auth_scope": "api://trapi/.default",
        "temperature":1.0,
        "max_tokens":4096,
    },
)

VLLM_PHI_4_SFT_APRIL_2025_CONFIG = ModelConfig(
    # Use this config if you have already deployed the model
    # and pass the service ports, num_servers, and model_name as commandline args
    LocalVLLMModel,
    {
        "temperature": 0.8,
        "max_tokens": 30000,
        "system_message": "Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <|dummy_86|> {Thought section} <|dummy_87|> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines:",
    }
)