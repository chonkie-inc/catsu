"""Main client for Catsu embedding API.

The Client class provides a unified interface for accessing multiple embedding
providers through a single API.
"""

import os
from typing import Any, Dict, List, Optional, Union

import httpx

from .catalog import ModelCatalog
from .providers import BaseProvider, registry
from .utils.errors import InvalidInputError, ModelNotFoundError


class Client:
    """Unified client for embedding APIs.

    Supports multiple embedding providers through a clean, consistent interface
    with built-in retry logic, cost tracking, and rich model metadata.

    Args:
        verbose: Enable verbose logging (default: False)
        max_retries: Maximum number of retry attempts (default: 3)
        timeout: Request timeout in seconds (default: 30)
        api_keys: Optional dict of API keys by provider name
                  (e.g., {"voyageai": "key123"})

    Example:
        >>> import catsu
        >>> client = catsu.Client(max_retries=3, timeout=30)
        >>>
        >>> # Three ways to specify provider:
        >>> # 1. Separate parameters
        >>> result = client.embed(
        ...     provider="voyageai",
        ...     model="voyage-3",
        ...     input="hello world"
        ... )
        >>>
        >>> # 2. Provider prefix
        >>> result = client.embed(
        ...     model="voyageai:voyage-3",
        ...     input="hello world"
        ... )
        >>>
        >>> # 3. Auto-detection (if model name is unique)
        >>> result = client.embed(model="voyage-3", input="hello world")

    """

    def __init__(
        self,
        verbose: bool = False,
        max_retries: int = 3,
        timeout: int = 30,
        api_keys: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize the Catsu client."""
        self.verbose = verbose
        self.max_retries = max_retries
        self.timeout = timeout
        self._api_keys = api_keys or {}

        # Initialize HTTP clients for sync and async
        self._http_client = httpx.Client(timeout=timeout)
        self._async_http_client = httpx.AsyncClient(timeout=timeout)

        # Provider registry
        self._providers: Dict[str, BaseProvider] = {}

        # Initialize model catalog
        self._catalog = ModelCatalog()

        # Load providers
        self._load_providers()

    def _load_providers(self) -> None:
        """Load and register available providers from registry."""
        for provider_name, provider_class in registry.items():
            self._providers[provider_name] = provider_class(
                http_client=self._http_client,
                async_http_client=self._async_http_client,
                api_key=self._get_api_key(provider_name),
                max_retries=self.max_retries,
                verbose=self.verbose,
            )

    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider from instance keys or environment.

        Args:
            provider: Provider name (e.g., "voyageai")

        Returns:
            API key if found, None otherwise

        """
        # Check instance-level keys first
        if provider in self._api_keys:
            return self._api_keys[provider]

        # Check environment variables
        env_var_map = {
            "voyageai": "VOYAGE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "cohere": "COHERE_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "jinaai": "JINA_API_KEY",
            "mistral": "MISTRAL_API_KEY",
        }

        env_var = env_var_map.get(provider)
        if env_var:
            return os.getenv(env_var)

        return None

    def _parse_model_string(
        self, model: str, provider: Optional[str] = None
    ) -> tuple[str, str]:
        """Parse model string to extract provider and model name.

        Supports three formats:
        1. provider:model (e.g., "voyageai:voyage-3")
        2. model with explicit provider param
        3. model with auto-detection

        Args:
            model: Model string (e.g., "voyage-3" or "voyageai:voyage-3")
            provider: Optional explicit provider name

        Returns:
            Tuple of (provider_name, model_name)

        Raises:
            AmbiguousModelError: If model name is ambiguous and no provider specified
            ModelNotFoundError: If model or provider not found

        """
        # Format 1: Check for provider prefix in model string
        if ":" in model:
            parsed_provider, parsed_model = model.split(":", 1)
            if provider and provider != parsed_provider:
                raise InvalidInputError(
                    f"Provider mismatch: '{provider}' specified but "
                    f"'{parsed_provider}' in model string",
                    parameter="provider",
                )
            return parsed_provider, parsed_model

        # Format 2: Explicit provider parameter
        if provider:
            return provider, model

        # Format 3: Auto-detection
        detected_provider = self._catalog.auto_detect_provider(model)
        if detected_provider:
            return detected_provider, model
        else:
            raise ModelNotFoundError(
                model=model,
                details={
                    "message": "Could not find model in any provider. "
                    "Please specify the provider explicitly."
                },
            )

    def _get_provider(self, provider_name: str) -> BaseProvider:
        """Get provider instance by name.

        Args:
            provider_name: Name of the provider (e.g., "voyageai")

        Returns:
            Provider instance

        Raises:
            ModelNotFoundError: If provider not found

        """
        if provider_name not in self._providers:
            raise ModelNotFoundError(
                model="",
                provider=provider_name,
                details={
                    "available_providers": list(self._providers.keys()),
                    "message": f"Provider '{provider_name}' not found",
                },
            )

        return self._providers[provider_name]

    def embed(
        self,
        model: str,
        input: Union[str, List[str]],
        provider: Optional[str] = None,
        input_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:  # -> EmbedResponse
        """Generate embeddings for input text(s).

        Args:
            model: Model name or "provider:model" string
            input: Single text string or list of text strings
            provider: Optional provider name (if not in model string)
            input_type: Optional input type hint ("query" or "document")
                       Used by some providers like VoyageAI
            **kwargs: Additional provider-specific parameters

        Returns:
            EmbedResponse with embeddings, usage, and metadata

        Raises:
            CatsuError: Base exception for all errors
            ProviderError: Provider-specific errors
            ModelNotFoundError: Model not found
            AmbiguousModelError: Model name is ambiguous

        Example:
            >>> client = Client()
            >>> result = client.embed(
            ...     model="voyage-3",
            ...     input="hello world",
            ...     provider="voyageai"
            ... )
            >>> print(result.embeddings)  # [[0.1, 0.2, ...]]
            >>> print(result.usage.total_tokens)  # 2
            >>> print(result.usage.total_cost)  # 0.0000002

        """
        # Parse provider and model
        provider_name, model_name = self._parse_model_string(model, provider)

        if self.verbose:
            print(f"Using provider: {provider_name}, model: {model_name}")

        # Get provider instance
        provider_instance = self._get_provider(provider_name)

        # Normalize input to list
        inputs = [input] if isinstance(input, str) else input

        # Call provider's embed method
        return provider_instance.embed(
            model=model_name,
            inputs=inputs,
            input_type=input_type,
            **kwargs,
        )

    async def aembed(
        self,
        model: str,
        input: Union[str, List[str]],
        provider: Optional[str] = None,
        input_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:  # -> EmbedResponse
        """Async version of embed().

        Generate embeddings for input text(s) asynchronously.

        Args:
            model: Model name or "provider:model" string
            input: Single text string or list of text strings
            provider: Optional provider name (if not in model string)
            input_type: Optional input type hint ("query" or "document")
            **kwargs: Additional provider-specific parameters

        Returns:
            EmbedResponse with embeddings, usage, and metadata

        Example:
            >>> import asyncio
            >>> client = Client()
            >>> result = await client.aembed(
            ...     model="voyage-3",
            ...     input="hello world",
            ...     provider="voyageai"
            ... )

        """
        # Parse provider and model
        provider_name, model_name = self._parse_model_string(model, provider)

        if self.verbose:
            print(f"Using provider: {provider_name}, model: {model_name}")

        # Get provider instance
        provider_instance = self._get_provider(provider_name)

        # Normalize input to list
        inputs = [input] if isinstance(input, str) else input

        # Call provider's aembed method
        return await provider_instance.aembed(
            model=model_name,
            inputs=inputs,
            input_type=input_type,
            **kwargs,
        )

    def list_models(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available models, optionally filtered by provider.

        Args:
            provider: Optional provider name to filter by

        Returns:
            List of model info dictionaries

        Example:
            >>> client = Client()
            >>> models = client.list_models(provider="voyageai")
            >>> for model in models:
            ...     print(f"{model['name']}: {model['dimensions']} dims")

        """
        models = self._catalog.list_models(provider=provider)
        # Convert ModelInfo objects to dictionaries
        return [model.model_dump() for model in models]

    def close(self) -> None:
        """Close HTTP clients and cleanup resources."""
        self._http_client.close()

    async def aclose(self) -> None:
        """Async version of close()."""
        await self._async_http_client.aclose()

    def __enter__(self) -> "Client":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    async def __aenter__(self) -> "Client":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.aclose()
