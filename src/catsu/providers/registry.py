"""Provider registry for Catsu.

Maps provider names to their implementation classes for dynamic loading.
"""

from typing import Dict, Type

from .base import BaseProvider
from .cloudflare import CloudflareProvider
from .cohere import CohereProvider
from .deepinfra import DeepInfraProvider
from .gemini import GeminiProvider
from .jinaai import JinaAIProvider
from .mistral import MistralProvider
from .mixedbread import MixedbreadProvider
from .nomic import NomicProvider
from .openai import OpenAIProvider
from .togetherai import TogetherAIProvider
from .voyageai import VoyageAIProvider

# Registry of available providers
registry: Dict[str, Type[BaseProvider]] = {
    "cloudflare": CloudflareProvider,
    "cohere": CohereProvider,
    "deepinfra": DeepInfraProvider,
    "gemini": GeminiProvider,
    "jinaai": JinaAIProvider,
    "mistral": MistralProvider,
    "mixedbread": MixedbreadProvider,
    "nomic": NomicProvider,
    "openai": OpenAIProvider,
    "togetherai": TogetherAIProvider,
    "voyageai": VoyageAIProvider,
}
