"""Catsu - A unified, batteries-included client for embedding APIs.

Catsu provides a clean, consistent interface for accessing multiple embedding
providers through a single API with built-in retry logic, cost tracking, and
rich model metadata.
"""

from .catalog import ModelCatalog
from .client import Client
from .models import EmbedResponse, ModelInfo, TokenizeResponse, Usage

__version__ = "0.0.1a2"
__author__ = "Chonkie, Inc."

__all__ = [
    "Client",
    "EmbedResponse",
    "Usage",
    "TokenizeResponse",
    "ModelInfo",
    "ModelCatalog",
    "__version__",
    "__author__",
]
