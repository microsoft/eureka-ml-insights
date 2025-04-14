""" This module contains config objects for the models used in the experiments. To use these configs, make sure to
replace the placeholders with your own keys.json file, secret key names, and endpint URLs where applicable. 
You can also add your custom models here by following the same pattern as the existing configs. """

from eureka_ml_insights.models import (
    AzureOpenAIO1Model,
    AzureOpenAIModel,
    ClaudeModel,
    DirectOpenAIModel,
    DirectOpenAIO1Model,
    GeminiModel,
    LlamaServerlessAzureRestEndpointModel,
    LLaVAHuggingFaceModel,
    LLaVAModel,
    MistralServerlessAzureRestEndpointModel,
    RestEndpointModel,
    TestModel,
    Phi4HFModel,
)

from .config import ModelConfig

# For models that require secret keys, you can store the keys in a json file and provide the path to the file
# in the secret_key_params dictionary. OR you can provide the key name and key vault URL to fetch the key from Azure Key Vault.
# You don't need to provide both the key_vault_url and local_keys_path. You can provide one of them based on your setup.


# Test model
TEST_MODEL_CONFIG = ModelConfig(TestModel, {})

# OpenAI models

OPENAI_SECRET_KEY_PARAMS = {
    "key_name": "openai", #"openai-models-west-us3", #"openai-models-australia-east", #"openai",
    "local_keys_path": "keys/aifeval-vault-azure-net.json",
    "key_vault_url": None,
}

OAI_O3_MINI_CONFIG = ModelConfig(
    DirectOpenAIO1Model,
    {
        "model_name": "o3-mini-2025-01-31",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

OAI_O1_20241217_CONFIG = ModelConfig(
    DirectOpenAIO1Model,
    {
        "model_name": "o1-2024-12-17",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

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

OAI_GPT4O_2024_11_20_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "model_name": "gpt-4o-2024-11-20",
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

OAI_GPT4O_MINI_2024_07_18_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "model_name": "gpt-4o-mini-2024-07-18",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

#################### TRAPI models ####################

#### TRAPI models ####


# Azure OAI models
## Azure OAI models -- TRAPI Models 
## https://dev.azure.com/msresearch/MSR%20Engineering/_wiki/wikis/MSR-Engineering.wiki/13498/Deployment-Model-Information

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

# this endpoint is used by all GCR members, only use when the AIF endpoints are down or not available 
TRAPI_GCR_SHARED_O1_CONFIG = ModelConfig(
    AzureOpenAIO1Model,
    {
        "url": "https://trapi.research.microsoft.com/gcr/shared",
        # o1 models only work with 2024-12-01-preview api version
        "api_version": '2024-12-01-preview',
        "model_name": "o1_2024-12-17",
        "auth_scope": "api://trapi/.default"
    },
)

TRAPI_O1_PREVIEW_CONFIG = ModelConfig(
    AzureOpenAIO1Model,
    {
        "url": "https://trapi.research.microsoft.com/msraif/shared",
        "api_version": '2024-10-21',
        "model_name": "o1-preview_2024-09-12",
        "auth_scope": "api://trapi/.default"
    },
)

TRAPI_GPT4O_2024_11_20_CONFIG = ModelConfig(
    AzureOpenAIModel,
    {
        "url": "https://trapi.research.microsoft.com/msraif/shared",
        "api_version": '2024-10-21',        
        "model_name": "gpt-4o_2024-11-20",
        "auth_scope": "api://trapi/.default",
        "temperature": 1.0,
        "max_tokens": 4096
    },
)



TRAPI_GPT4O_2024_05_13_CONFIG = ModelConfig(
    AzureOpenAIModel,
    {
        "url": "https://trapi.research.microsoft.com/msraif/shared",
        "api_version": '2024-10-21',
        # "model_name": "gpt-4o_2024-08-06",
        "model_name": "gpt-4o_2024-11-20",
        "auth_scope": "api://trapi/.default",
        "temperature": 1.0,
        "max_tokens": 4096
    },
)

TRAPI_GPT4_VISION_PREVIEW_CONFIG = ModelConfig(
    AzureOpenAIModel,
    {
        "url": "https://trapi.research.microsoft.com/msraif/shared",
        "model_name": "gpt-4_vision-preview",
        "auth_scope": "api://trapi/.default"
    },
)

TRAPI_GPT4V_TURBO_2024_04_09_CONFIG = ModelConfig(
    AzureOpenAIModel,
    {
        "url": "https://trapi.research.microsoft.com/msraif/shared",
        "model_name": "gpt-4_turbo-2024-04-09",
        "api_version": '2024-10-21',
        "auth_scope": "api://trapi/.default"
    },
)

TRAPI_GPT4_1106_PREVIEW_CONFIG = ModelConfig(
    AzureOpenAIModel,
    {
        "url": "https://trapi.research.microsoft.com/msraif/shared",
        "model_name": "gpt-4_1106-Preview",
        "auth_scope": "api://trapi/.default"
    },
)


TRAPI_AIF_O3_MINI_CONFIG = ModelConfig(
    AzureOpenAIO1Model,
    {
        "url": "https://trapi.research.microsoft.com/msraif/shared",
        # o1 models only work with > 2024-12-01-preview api version
        "api_version": '2025-01-01-preview',
        "model_name": "o3-mini_2025-01-31",
        "reasoning_effort": "high",
        "auth_scope": "api://trapi/.default"
    },
)



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


### claude models

# Claude models  
CLAUDE_SECRET_KEY_PARAMS = {
    "key_name": "aif-eval-claude",
    "local_keys_path": "keys/aifeval-vault-azure-net.json",
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
        "model_name": "claude-3-5-sonnet-20241022",
        "temperature": 1.0,
        "max_tokens": 4096
    },
)


# # Claude models
# CLAUDE_SECRET_KEY_PARAMS = {
#     "key_name": "your_claude_secret_key_name",
#     "local_keys_path": "keys/keys.json",
#     "key_vault_url": None,
# }

# CLAUDE_3_OPUS_CONFIG = ModelConfig(
#     ClaudeModel,
#     {
#         "model_name": "claude-3-opus-20240229",
#         "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
#     },
# )

# CLAUDE_3_5_SONNET_CONFIG = ModelConfig(
#     ClaudeModel,
#     {
#         "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
#         "model_name": "claude-3-5-sonnet-20240620",
#     },
# )

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

##################################

AIF_NT_LLAMA3_1_405B_INSTRUCT_EASTUS_OSS_CONFIG_2 = ModelConfig(
    LlamaServerlessAzureRestEndpointModel,
    {
        "url": "https://Meta-Llama-3-1-405B-Instruct-2.eastus.models.ai.azure.com/v1/chat/completions",
        "secret_key_params": {
            "key_name": "aif-nt-meta-llama-3-1-405b-instruct-2-oss",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
            # "temperature": 1.0,
            # "max_tokens": 4096
        },
    },
)

AIF_NT_LLAMA3_1_405B_INSTRUCT_WESTUS_CONFIG = ModelConfig(
    LlamaServerlessAzureRestEndpointModel,
    {
        "url": "https://Meta-Llama-3-1-405B-Instruct-4.westus.models.ai.azure.com/v1/chat/completions",
        "secret_key_params": {
            "key_name": "aif-nt-meta-llama-3-1-405b-instruct-4-westus",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
    },
)

AIF_NT_LLAMA3_1_405B_INSTRUCT_WESTUS3_CONFIG = ModelConfig(
    LlamaServerlessAzureRestEndpointModel,
    {
        "url": "https://Meta-Llama-3-1-405B-Instruct-2.westus3.models.ai.azure.com/v1/chat/completions",
        "secret_key_params": {
            "key_name": "aif-nt-meta-llama-3-1-405b-instruct-westus3",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
    },
)


##############################

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

################################

#################################

# Phi models
AIF_PHI4_CONFIG_1 = ModelConfig(
    RestEndpointModel,
    {
        "url": "https://aif-phi4.eastus2.inference.ml.azure.com/score",
        "secret_key_params": {
            "key_name": "phi-4-aif",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
        "temperature": 1.0,
        "max_tokens": 4096
    },
)

AIF_PHI4_CONFIG_2 = ModelConfig(
    RestEndpointModel,
    {
        "url": "https://aif-phi4-2.eastus2.inference.ml.azure.com/score",
        "secret_key_params": {
            "key_name": "phi-4-aif-2",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
        "temperature": 1.0,
        "max_tokens": 4096
    },
)

GCR_PHI4_CONFIG = ModelConfig(
    RestEndpointModel,
    {
        "url": "https://gcr-phi-4.westus3.inference.ml.azure.com/score",
        "secret_key_params": {
            "key_name": "gcr-phi-4",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
        "temperature": 1.0,
        "max_tokens": 4096
    },
)

GCR_PHI3_MINI_128K_INSTRUCT_CONFIG = ModelConfig(
    RestEndpointModel,
    {
        "url": "https://gcr-phi3-mini-128k-instruct.westus3.inference.ml.azure.com/score",
        "secret_key_params": {
            "key_name": "phi-3-mini-128k-instruct-7",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
    },
)

PHI4HF_CONFIG = ModelConfig(
    Phi4HFModel,
    {"model_name": "microsoft/phi-4"},
)

