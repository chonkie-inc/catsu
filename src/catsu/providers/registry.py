"""Provider registry for Catsu.

Maps provider names to their implementation classes for dynamic loading.
"""

from typing import Dict, Type

from .base import BaseProvider
from .cohere import CohereProvider
from .gemini import GeminiProvider
from .jinaai import JinaAIProvider
from .mistral import MistralProvider
from .nomic import NomicProvider
from .openai import OpenAIProvider
from .voyageai import VoyageAIProvider

# Registry of available providers
registry: Dict[str, Type[BaseProvider]] = {
    "cohere": CohereProvider,
    "gemini": GeminiProvider,
    "jinaai": JinaAIProvider,
    "mistral": MistralProvider,
    "nomic": NomicProvider,
    "openai": OpenAIProvider,
    "voyageai": VoyageAIProvider,
}
