"""Provider registry for Mimie.

Maps provider names to their implementation classes for dynamic loading.
"""

from typing import Dict, Type

from .base import BaseProvider
from .voyageai import VoyageAIProvider

# Registry of available providers
registry: Dict[str, Type[BaseProvider]] = {
    "voyageai": VoyageAIProvider,
}
