"""Provider registry for Catsu.

Maps provider names to their implementation classes for dynamic loading.
"""

from typing import Dict, Type

from .base import BaseProvider
from .cohere import CohereProvider
from .openai import OpenAIProvider
from .voyageai import VoyageAIProvider

# Registry of available providers
registry: Dict[str, Type[BaseProvider]] = {
    "cohere": CohereProvider,
    "openai": OpenAIProvider,
    "voyageai": VoyageAIProvider,
}
