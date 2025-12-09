"""
Mimie - A unified, batteries-included client for embedding APIs.

Mimie provides a clean, consistent interface for accessing multiple embedding
providers through a single API with built-in retry logic, cost tracking, and
rich model metadata.
"""

from .client import Client
from .models import EmbedResponse, ModelInfo, TokenizeResponse, Usage
from .catalog import ModelCatalog

__version__ = "0.1.0"
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
