""" This module contains config objects for the models used in the experiments. To use these configs, make sure to
replace the placeholders with your own keys.json file, secret key names, and endpint URLs where applicable. 
You can also add your custom models here by following the same pattern as the existing configs. """

from eureka_ml_insights.models import (
    AzureOpenAIOModel,
    ClaudeModel,
    ClaudeReasoningModel,
    DirectOpenAIModel,
    DirectOpenAIOModel,
    GeminiModel,
    LlamaServerlessAzureRestEndpointModel,
    LLaVAHuggingFaceModel,
    LLaVAModel,
    LocalVLLMModel,
    Phi4HFModel,
    MistralServerlessAzureRestEndpointModel,
    DeepseekR1ServerlessAzureRestEndpointModel,
    RestEndpointModel,
    TogetherModel,
    TestModel,
    OfflineFileModel
)
from eureka_ml_insights.models.models import AzureOpenAIModel

from .config import ModelConfig

AIF_NT_DEEPSEEK_R1_CONFIG = ModelConfig(
    DeepseekR1ServerlessAzureRestEndpointModel,
    {
        "url": "https://DeepSeek-R1-aif-nt.eastus.models.ai.azure.com/v1/chat/completions",
        "secret_key_params": {
            "key_name": "aif-nt-deepseek-r1",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
        "max_tokens": 32768,
        "timeout": 1200
    },
)

# For models that require secret keys, you can store the keys in a json file and provide the path to the file
# in the secret_key_params dictionary. OR you can provide the key name and key vault URL to fetch the key from Azure Key Vault.
# You don't need to provide both the key_vault_url and local_keys_path. You can provide one of them based on your setup.
GATEWAY_SECRET_KEY_PARAMS = {
    "key_name": "gateway",
    "local_keys_path": "/home/sayouse/git/eureka-ml-insights/bingbong/aifeval-vault-azure-net.json",
    "key_vault_url": None,
}

GATEWAY_GPT_4O_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "base_url": "https://gateway.phyagi.net/api/",
        "model_name": "gpt-4o",
        "secret_key_params": GATEWAY_SECRET_KEY_PARAMS,
#        "extra_body":{"tier": "impact", "cache_ttl": 0},
        "temperature": 0,
        "max_tokens": 16000,
        "top_p": 0.95,
    },
)

# Test model
TEST_MODEL_CONFIG = ModelConfig(TestModel, {})

OFFLINE_MODEL_CONFIG = ModelConfig(
    OfflineFileModel,
    {
        "model_name": "Teacher_Agent_V1",
        # This file contains the offline results from a model or agentic system
        # The file should contain at least the following fields:
        # "model_output", "prompt", and "data_repeat_id" for experiments that have several runs/repeats
        "file_path": r"your_offline_model_results.jsonl",
    },
)

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

TOGETHER_DEEPSEEK_R1_Distill_Llama_70B_CONFIG = ModelConfig(
    TogetherModel,
    {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "secret_key_params": TOGETHER_SECRET_KEY_PARAMS,
        "temperature": 0.6,
        # high max token limit for deep seek
        # otherwise the answers may be cut in the middle
        "max_tokens": 65536
    },
)

# OpenAI models
OPENAI_SECRET_KEY_PARAMS = {
    "key_name": "openai",
    "local_keys_path": "/home/sayouse/git/eureka-ml-insights/bingbong/aifeval-vault-azure-net.json",
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

OAI_O1_PREVIEW_AZURE_CONFIG = ModelConfig(
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

# Gemini models
GEMINI_SECRET_KEY_PARAMS = {
    "key_name": "your_gemini_secret_key_name",
    "local_keys_path": "keys/keys.json",
    "key_vault_url": None,
}

GEMINI_V2_FLASH_THINKING_EXP_0121_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-2.0-flash-thinking-exp-01-21",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
	    "max_tokens": 32768
    },
)

GEMINI_V2_PRO_EXP_0205_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-2.0-pro-exp-02-05",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
    },
)

GEMINI_V15_PRO_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-1.5-pro",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
    },
)

# Claude models
CLAUDE_SECRET_KEY_PARAMS = {
    "key_name": "your_claude_secret_key_name",
    "local_keys_path": "keys/keys.json",
    "key_vault_url": None,
}

CLAUDE_3_7_SONNET_THINKING_CONFIG = ModelConfig(
    ClaudeReasoningModel,
    {
        "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
        "model_name": "claude-3-7-sonnet-20250219",
        "thinking_enabled": True,
        "thinking_budget": 30720,
        "max_tokens": 32768, # This number should always be higher than the thinking budget
        "temperature": 1.0, # As of 03/08/2025, thinking only works with temperature 1.0
        "timeout": 600, # We set a timeout of 10 minutes for thinking
    },
)

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

CLAUDE_3_7_SONNET_CONFIG = ModelConfig(
    ClaudeModel,
    {
        "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
        "model_name": "claude-3-7-sonnet-20250219",
    },
)

# The config below uses temperature 1.0 to enable more diverse outputs
CLAUDE_3_5_SONNET_20241022_TEMP1_CONFIG = ModelConfig(
    ClaudeModel,
    {
        "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
        "model_name": "claude-3-5-sonnet-20241022",
		"temperature": 1.0,
        "max_tokens": 4096
    },
)

CLAUDE_3_5_SONNET_20241022_CONFIG = ModelConfig(
    ClaudeModel,
    {
        "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
        "model_name": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096
    },
)

#  LLAVA models
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

# Local VLLM Models
# Adapt to your local deployments, or give enough info for vllm deployment.
PHI4_LOCAL_CONFIG = ModelConfig(
    LocalVLLMModel,
    {
        # this name must match the vllm deployment name/path
        "model_name": "microsoft/phi-4",
        # specify ports in case the model is already deployed
        "ports": ["8002", "8003"],
    },
)
QWQ32B_LOCAL_CONFIG = ModelConfig(
    LocalVLLMModel,
    {
        # this name must match the vllm deployment name/path
        "model_name": "Qwen/QwQ-32B",
        # certain args will get passed to the vllm serve command
        "tensor_parallel_size": 2,
    },
)

# DeepSeek R1 Endpoints on Azure
DEEPSEEK_R1_CONFIG = ModelConfig(
    DeepseekR1ServerlessAzureRestEndpointModel,
    {
        "url": "your/endpoint/url",
        "secret_key_params": {
            "key_name": "your_deepseek_r1_secret_key_name",
            "local_keys_path": "keys/keys.json",
            "key_vault_url": None,
        },
        "max_tokens": 32768,
        # the timeout parameter is passed to urllib.request.urlopen(request, timeout=self.timeout) in ServerlessAzureRestEndpointModel
        "timeout": 600,
    },
)

VLLM_DEEPSEEK_CONFIG = ModelConfig(
    # Use this config if you have already deployed the model
    # and pass the service ports, num_servers, and model_name as commandline args
    LocalVLLMModel,
    {
        "temperature": 0.6,
        "max_tokens": 30000,
    }
)

VLLM_QWEN3_CONFIG = ModelConfig(
    # Use this config if you have already deployed the model
    # and pass the service ports, num_servers, and model_name as commandline args
    LocalVLLMModel,
    {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 32768,
    }
)
