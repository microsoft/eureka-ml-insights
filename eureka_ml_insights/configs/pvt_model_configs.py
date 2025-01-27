""" This module contains config objects for the models used in the experiments. To use these configs, make sure to
replace the placeholders with your own keys.json file, secret key names, and endpint URLs where applicable. 
You can also add your custom models here by following the same pattern as the existing configs. """

from eureka_ml_insights.models import (
    AzureOpenAIO1Model,
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
)
from eureka_ml_insights.models.models import AzureOpenAIModel

from .config import ModelConfig

# include any private model configs and keys you require here. this file is gitignored