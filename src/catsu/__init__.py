"""Catsu - A unified, batteries-included client for embedding APIs.

Catsu provides a clean, consistent interface for accessing multiple embedding
providers through a single API with built-in retry logic, cost tracking, and
rich model metadata.
"""

from .catalog import ModelCatalog
from .client import Client
from .models import EmbedResponse, ModelInfo, TokenizeResponse, Usage
from .utils.errors import (
    AuthenticationError,
    CatsuError,
    ConfigurationError,
    FallbackExhaustedError,
    InvalidInputError,
    ModelNotFoundError,
    NetworkError,
    ProviderError,
    RateLimitError,
    TimeoutError,
    UnsupportedFeatureError,
)

__version__ = "0.0.2"
__author__ = "Chonkie, Inc."

__all__ = [
    # Client and models
    "Client",
    "EmbedResponse",
    "Usage",
    "TokenizeResponse",
    "ModelInfo",
    "ModelCatalog",
    # Exceptions
    "CatsuError",
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
    "TimeoutError",
    "NetworkError",
    "ModelNotFoundError",
    "InvalidInputError",
    "UnsupportedFeatureError",
    "ConfigurationError",
    "FallbackExhaustedError",
    # Package metadata
    "__version__",
    "__author__",
]
