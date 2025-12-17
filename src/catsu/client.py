"""Main client for Catsu embedding API.

The Client class provides a unified interface for accessing multiple embedding
providers through a single API.
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import httpx

from .catalog import ModelCatalog
from .models import EmbedResponse, TokenizeResponse
from .providers import BaseProvider, registry
from .utils.errors import (
    AuthenticationError,
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

# Type aliases for fallback configuration
FallbackConfig = Dict[str, Any]  # {"model": "...", **provider_overrides}
Fallbacks = Union[str, FallbackConfig, List[Union[str, FallbackConfig]], None]


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
        fallbacks: Fallbacks = None,
        allow_unsafe_fallback: bool = False,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
    ) -> None:
        """Initialize the Catsu client.

        Args:
            verbose: Enable verbose logging (default: False)
            max_retries: Maximum number of retry attempts (default: 3)
            timeout: Request timeout in seconds (default: 30)
            api_keys: Optional dict of API keys by provider name
            fallbacks: Default fallback models for all requests
            allow_unsafe_fallback: If True, allow fallback to models with
                different dimensions (default: False, only dimension-compatible)
            base_delay: Base delay between retry cycles in seconds
            max_delay: Maximum delay between retry cycles in seconds

        """
        self.verbose = verbose
        self.max_retries = max_retries
        self.timeout = timeout
        self._api_keys = api_keys or {}

        # Fallback configuration
        self.fallbacks = fallbacks
        self.allow_unsafe_fallback = allow_unsafe_fallback
        self.base_delay = base_delay
        self.max_delay = max_delay

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
                catalog=self._catalog,
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
            "nomic": "NOMIC_API_KEY",
            "cloudflare": "CLOUDFLARE_API_KEY",
            "deepinfra": "DEEPINFRA_API_KEY",
            "mixedbread": "MIXEDBREAD_API_KEY",
            "togetherai": "TOGETHERAI_API_KEY",
        }

        env_var = env_var_map.get(provider)
        if env_var:
            return os.getenv(env_var)

        return None

    def _parse_model_string(
        self, model: str, provider: Optional[str] = None
    ) -> Tuple[str, str]:
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

    # -------------------------------------------------------------------------
    # Fallback helper methods
    # -------------------------------------------------------------------------

    def _normalize_fallbacks(self, fallbacks: Fallbacks) -> List[FallbackConfig]:
        """Convert any fallback input format to List[FallbackConfig].

        Args:
            fallbacks: Fallback specification (str, dict, list, or None)

        Returns:
            Normalized list of fallback configurations

        """
        if fallbacks is None:
            return []
        if isinstance(fallbacks, str):
            return [{"model": fallbacks}]
        if isinstance(fallbacks, dict):
            return [fallbacks]
        # List - normalize each item
        return [{"model": f} if isinstance(f, str) else f for f in fallbacks]

    def _get_env_var_for_provider(self, provider: str) -> str:
        """Get the environment variable name for a provider's API key."""
        env_var_map = {
            "voyageai": "VOYAGE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "cohere": "COHERE_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "jinaai": "JINA_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "nomic": "NOMIC_API_KEY",
            "cloudflare": "CLOUDFLARE_API_KEY",
            "deepinfra": "DEEPINFRA_API_KEY",
            "mixedbread": "MIXEDBREAD_API_KEY",
            "togetherai": "TOGETHERAI_API_KEY",
        }
        return env_var_map.get(provider, f"{provider.upper()}_API_KEY")

    def _validate_fallback_credentials(
        self,
        primary_model: str,
        fallback_configs: List[FallbackConfig],
    ) -> None:
        """Validate API keys exist for primary and all fallback models.

        Args:
            primary_model: The primary model string
            fallback_configs: Normalized fallback configurations

        Raises:
            ConfigurationError: If any required API keys are missing

        """
        all_models = [primary_model] + [f["model"] for f in fallback_configs]
        missing = []

        for model in all_models:
            provider_name, _ = self._parse_model_string(model)
            api_key = self._get_api_key(provider_name)
            if not api_key:
                env_var = self._get_env_var_for_provider(provider_name)
                missing.append({"model": model, "provider": provider_name, "env_var": env_var})

        if missing:
            details = "\n".join(
                f"  - {m['model']} ({m['provider']}): set {m['env_var']}"
                for m in missing
            )
            raise ConfigurationError(
                f"Missing API keys for models:\n{details}",
                details={"missing_credentials": missing},
            )

    def _get_target_dimensions(
        self,
        provider_name: str,
        model_name: str,
        kwargs: Dict[str, Any],
    ) -> int:
        """Get the target dimensions (from kwargs or model default).

        Args:
            provider_name: Provider name
            model_name: Model name
            kwargs: Request kwargs that may contain dimensions

        Returns:
            Target dimensions value

        """
        if "dimensions" in kwargs and kwargs["dimensions"] is not None:
            return kwargs["dimensions"]
        model_info = self._catalog.get_model_info(provider_name, model_name)
        return model_info.dimensions

    def _can_produce_dimensions(
        self,
        provider_name: str,
        model_name: str,
        target_dims: int,
    ) -> bool:
        """Check if a model can produce the target dimensions.

        Args:
            provider_name: Provider name
            model_name: Model name
            target_dims: Required dimensions

        Returns:
            True if model can produce target dimensions

        """
        model_info = self._catalog.get_model_info(provider_name, model_name)

        # If model supports custom dimensions, it can produce any size
        # (we assume the user knows valid ranges)
        if model_info.supports_dimensions:
            return True

        # Otherwise, model must have matching default dimensions
        return model_info.dimensions == target_dims

    def _filter_compatible_fallbacks(
        self,
        target_dims: int,
        fallback_configs: List[FallbackConfig],
    ) -> List[FallbackConfig]:
        """Filter fallbacks to only those that can produce target dimensions.

        Args:
            target_dims: Required dimensions
            fallback_configs: Normalized fallback configurations

        Returns:
            Filtered list of compatible fallback configurations

        """
        compatible = []

        for config in fallback_configs:
            model = config["model"]
            provider_name, model_name = self._parse_model_string(model)

            if self._can_produce_dimensions(provider_name, model_name, target_dims):
                # If model supports custom dims and none specified, add target
                model_info = self._catalog.get_model_info(provider_name, model_name)
                if model_info.supports_dimensions and "dimensions" not in config:
                    config = {**config, "dimensions": target_dims}
                compatible.append(config)
            elif self.verbose:
                print(f"[catsu] Skipping {model}: incompatible dimensions")

        return compatible

    def _is_fallback_error(self, error: Exception) -> bool:
        """Check if an error should trigger fallback.

        Fallback is triggered for transient errors (rate limits, timeouts,
        network issues, server errors). Client errors (auth, invalid input)
        should not trigger fallback.

        Args:
            error: The exception to check

        Returns:
            True if fallback should be attempted

        """
        # These errors should trigger fallback
        if isinstance(error, (RateLimitError, TimeoutError, NetworkError)):
            return True

        # Provider errors with 5xx status codes should trigger fallback
        if isinstance(error, ProviderError):
            if error.status_code and error.status_code >= 500:
                return True

        # Auth errors and client errors should NOT trigger fallback
        if isinstance(error, (AuthenticationError, InvalidInputError)):
            return False

        # Provider errors with 4xx should NOT trigger fallback
        if isinstance(error, ProviderError):
            if error.status_code and 400 <= error.status_code < 500:
                return False

        # Default: don't fallback for unknown errors
        return False

    # -------------------------------------------------------------------------
    # Fallback orchestration
    # -------------------------------------------------------------------------

    def _embed_with_fallbacks(
        self,
        primary_model: str,
        inputs: List[str],
        fallback_configs: List[FallbackConfig],
        allow_unsafe: bool,
        input_type: Optional[Literal["query", "document"]] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Execute embed with fallback chain using round-robin retries.

        Args:
            primary_model: Primary model string
            inputs: List of input texts
            fallback_configs: Normalized fallback configurations
            allow_unsafe: Whether to allow dimension-incompatible fallbacks
            input_type: Optional input type hint
            dimensions: Optional output dimensions
            api_key: Optional API key override
            **kwargs: Additional provider-specific parameters

        Returns:
            EmbedResponse from successful model

        Raises:
            FallbackExhaustedError: If all models fail after all cycles

        """
        # 1. Validate all credentials upfront (fail fast)
        self._validate_fallback_credentials(primary_model, fallback_configs)

        # 2. Get target dimensions from primary model
        primary_provider, primary_model_name = self._parse_model_string(primary_model)
        target_dims = self._get_target_dimensions(
            primary_provider, primary_model_name, {"dimensions": dimensions}
        )

        # 3. Filter by dimension compatibility if safe mode
        if not allow_unsafe:
            fallback_configs = self._filter_compatible_fallbacks(target_dims, fallback_configs)

        # 4. Build execution chain: [primary, ...fallbacks]
        chain: List[FallbackConfig] = [
            {"model": primary_model, "dimensions": dimensions, "api_key": api_key, **kwargs}
        ]
        for config in fallback_configs:
            chain.append({**config})

        errors: Dict[str, Exception] = {}
        cycle = 1

        # 5. Round-robin with global retries
        while cycle <= self.max_retries:
            if self.verbose:
                print(f"[catsu] Starting cycle {cycle}/{self.max_retries}")

            # Try each model in the chain (1 attempt each)
            for config in chain:
                model = config["model"]
                model_kwargs = {k: v for k, v in config.items() if k != "model"}

                try:
                    # Parse model and get provider
                    provider_name, model_name = self._parse_model_string(model)
                    model_info = self._catalog.get_model_info(provider_name, model_name)

                    # Validate features for this model
                    model_dims = model_kwargs.get("dimensions")
                    if model_dims is not None and not model_info.supports_dimensions:
                        # Skip this model if it doesn't support requested dimensions
                        if self.verbose:
                            print(f"[catsu] Skipping {model}: doesn't support custom dimensions")
                        continue

                    model_input_type = input_type
                    if model_input_type is not None and not model_info.supports_input_type:
                        # Use model without input_type
                        model_input_type = None

                    # Get provider and make request
                    provider_instance = self._get_provider(provider_name)
                    response = provider_instance.embed(
                        model=model_name,
                        inputs=inputs,
                        input_type=model_input_type,
                        dimensions=model_dims,
                        api_key=model_kwargs.get("api_key"),
                        **{k: v for k, v in model_kwargs.items() if k not in ("dimensions", "api_key")},
                    )

                    # Add fallback metadata if not primary
                    if model != primary_model:
                        response.requested_model = primary_model
                        response.fallback_used = True
                        primary_error = errors.get(primary_model)
                        response.fallback_reason = (
                            type(primary_error).__name__ if primary_error else "Unknown"
                        )

                    if self.verbose and model != primary_model:
                        print(f"[catsu] Fallback succeeded with {model}")

                    return response

                except Exception as e:
                    errors[model] = e
                    if self.verbose:
                        print(f"[catsu] Cycle {cycle}: {model} failed: {e}")

                    # Check if this error should trigger fallback
                    if not self._is_fallback_error(e):
                        # Re-raise non-fallback errors immediately
                        raise

                    # Continue to next model in chain

            # All models failed this cycle - backoff before next cycle
            if cycle < self.max_retries:
                backoff = min(
                    self.base_delay * (2 ** (cycle - 1)),
                    self.max_delay,
                )
                if self.verbose:
                    print(f"[catsu] All models failed cycle {cycle}, backing off {backoff}s...")
                time.sleep(backoff)

            cycle += 1

        # 6. All cycles exhausted
        raise FallbackExhaustedError(
            primary_model=primary_model,
            fallbacks=[f["model"] for f in fallback_configs],
            errors=errors,
            cycles_attempted=self.max_retries,
        )

    async def _aembed_with_fallbacks(
        self,
        primary_model: str,
        inputs: List[str],
        fallback_configs: List[FallbackConfig],
        allow_unsafe: bool,
        input_type: Optional[Literal["query", "document"]] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Async version of _embed_with_fallbacks."""
        # 1. Validate all credentials upfront (fail fast)
        self._validate_fallback_credentials(primary_model, fallback_configs)

        # 2. Get target dimensions from primary model
        primary_provider, primary_model_name = self._parse_model_string(primary_model)
        target_dims = self._get_target_dimensions(
            primary_provider, primary_model_name, {"dimensions": dimensions}
        )

        # 3. Filter by dimension compatibility if safe mode
        if not allow_unsafe:
            fallback_configs = self._filter_compatible_fallbacks(target_dims, fallback_configs)

        # 4. Build execution chain: [primary, ...fallbacks]
        chain: List[FallbackConfig] = [
            {"model": primary_model, "dimensions": dimensions, "api_key": api_key, **kwargs}
        ]
        for config in fallback_configs:
            chain.append({**config})

        errors: Dict[str, Exception] = {}
        cycle = 1

        # 5. Round-robin with global retries
        while cycle <= self.max_retries:
            if self.verbose:
                print(f"[catsu] Starting cycle {cycle}/{self.max_retries}")

            # Try each model in the chain (1 attempt each)
            for config in chain:
                model = config["model"]
                model_kwargs = {k: v for k, v in config.items() if k != "model"}

                try:
                    # Parse model and get provider
                    provider_name, model_name = self._parse_model_string(model)
                    model_info = self._catalog.get_model_info(provider_name, model_name)

                    # Validate features for this model
                    model_dims = model_kwargs.get("dimensions")
                    if model_dims is not None and not model_info.supports_dimensions:
                        if self.verbose:
                            print(f"[catsu] Skipping {model}: doesn't support custom dimensions")
                        continue

                    model_input_type = input_type
                    if model_input_type is not None and not model_info.supports_input_type:
                        model_input_type = None

                    # Get provider and make async request
                    provider_instance = self._get_provider(provider_name)
                    response = await provider_instance.aembed(
                        model=model_name,
                        inputs=inputs,
                        input_type=model_input_type,
                        dimensions=model_dims,
                        api_key=model_kwargs.get("api_key"),
                        **{k: v for k, v in model_kwargs.items() if k not in ("dimensions", "api_key")},
                    )

                    # Add fallback metadata if not primary
                    if model != primary_model:
                        response.requested_model = primary_model
                        response.fallback_used = True
                        primary_error = errors.get(primary_model)
                        response.fallback_reason = (
                            type(primary_error).__name__ if primary_error else "Unknown"
                        )

                    if self.verbose and model != primary_model:
                        print(f"[catsu] Fallback succeeded with {model}")

                    return response

                except Exception as e:
                    errors[model] = e
                    if self.verbose:
                        print(f"[catsu] Cycle {cycle}: {model} failed: {e}")

                    if not self._is_fallback_error(e):
                        raise

            # All models failed this cycle - backoff before next cycle
            if cycle < self.max_retries:
                backoff = min(
                    self.base_delay * (2 ** (cycle - 1)),
                    self.max_delay,
                )
                if self.verbose:
                    print(f"[catsu] All models failed cycle {cycle}, backing off {backoff}s...")
                await asyncio.sleep(backoff)

            cycle += 1

        # 6. All cycles exhausted
        raise FallbackExhaustedError(
            primary_model=primary_model,
            fallbacks=[f["model"] for f in fallback_configs],
            errors=errors,
            cycles_attempted=self.max_retries,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def embed(
        self,
        model: str,
        input: Union[str, List[str]],
        provider: Optional[str] = None,
        input_type: Optional[Literal["query", "document"]] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        fallbacks: Fallbacks = None,
        allow_unsafe_fallback: Optional[bool] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings for input text(s).

        Args:
            model: Model name or "provider:model" string
            input: Single text string or list of text strings
            provider: Optional provider name (if not in model string)
            input_type: Optional input type hint ("query" or "document")
                       Used by some providers like VoyageAI
            dimensions: Optional output dimensions (model must support this feature)
            api_key: Optional API key override for this specific request
            fallbacks: Fallback model(s) to try if primary fails. Can be:
                - str: Single model name (e.g., "text-embedding-3-small")
                - dict: Model with config (e.g., {"model": "...", "api_key": "..."})
                - list: Multiple fallbacks in order of preference
            allow_unsafe_fallback: If True, allow fallback to models with different
                dimensions (overrides client-level setting)
            **kwargs: Additional provider-specific parameters

        Returns:
            EmbedResponse with embeddings, usage, and metadata

        Raises:
            CatsuError: Base exception for all errors
            ProviderError: Provider-specific errors
            ModelNotFoundError: Model not found
            AmbiguousModelError: Model name is ambiguous
            UnsupportedFeatureError: Model doesn't support requested feature
            ConfigurationError: Missing API keys for fallback models
            FallbackExhaustedError: All fallback models failed

        Example:
            >>> client = Client()
            >>> result = client.embed(
            ...     model="voyage-3",
            ...     input="hello world",
            ...     fallbacks=["text-embedding-3-small", "embed-english-v3.0"]
            ... )
            >>> print(result.embeddings)  # [[0.1, 0.2, ...]]
            >>> print(result.fallback_used)  # True if fallback was used

        """
        # Normalize input to list
        inputs = [input] if isinstance(input, str) else input

        # Determine effective fallbacks (per-request overrides client-level)
        effective_fallbacks = fallbacks if fallbacks is not None else self.fallbacks
        effective_allow_unsafe = (
            allow_unsafe_fallback
            if allow_unsafe_fallback is not None
            else self.allow_unsafe_fallback
        )

        # If fallbacks are configured, use fallback logic
        if effective_fallbacks is not None:
            # Construct full model string with provider if specified
            full_model = f"{provider}:{model}" if provider and ":" not in model else model

            fallback_configs = self._normalize_fallbacks(effective_fallbacks)
            return self._embed_with_fallbacks(
                primary_model=full_model,
                inputs=inputs,
                fallback_configs=fallback_configs,
                allow_unsafe=effective_allow_unsafe,
                input_type=input_type,
                dimensions=dimensions,
                api_key=api_key,
                **kwargs,
            )

        # No fallbacks - use original logic
        # Parse provider and model
        provider_name, model_name = self._parse_model_string(model, provider)

        if self.verbose:
            print(f"Using provider: {provider_name}, model: {model_name}")

        # Get model info for feature validation
        model_info = self._catalog.get_model_info(provider_name, model_name)

        # Validate dimensions support if dimensions is provided
        if dimensions is not None:
            if not model_info.supports_dimensions:
                raise UnsupportedFeatureError(
                    f"Model '{model_name}' does not support custom dimensions",
                    model=model_name,
                    provider=provider_name,
                    feature="dimensions",
                )

        # Validate input_type support if input_type is provided
        if input_type is not None:
            if not model_info.supports_input_type:
                raise UnsupportedFeatureError(
                    f"Model '{model_name}' does not support input_type parameter",
                    model=model_name,
                    provider=provider_name,
                    feature="input_type",
                )

        # Get provider instance
        provider_instance = self._get_provider(provider_name)

        # Call provider's embed method
        return provider_instance.embed(
            model=model_name,
            inputs=inputs,
            input_type=input_type,
            dimensions=dimensions,
            api_key=api_key,
            **kwargs,
        )

    async def aembed(
        self,
        model: str,
        input: Union[str, List[str]],
        provider: Optional[str] = None,
        input_type: Optional[Literal["query", "document"]] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        fallbacks: Fallbacks = None,
        allow_unsafe_fallback: Optional[bool] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Async version of embed().

        Generate embeddings for input text(s) asynchronously.

        Args:
            model: Model name or "provider:model" string
            input: Single text string or list of text strings
            provider: Optional provider name (if not in model string)
            input_type: Optional input type hint ("query" or "document")
            dimensions: Optional output dimensions (model must support this feature)
            api_key: Optional API key override for this specific request
            fallbacks: Fallback model(s) to try if primary fails
            allow_unsafe_fallback: If True, allow fallback to models with different
                dimensions (overrides client-level setting)
            **kwargs: Additional provider-specific parameters

        Returns:
            EmbedResponse with embeddings, usage, and metadata

        Raises:
            UnsupportedFeatureError: Model doesn't support requested feature
            ConfigurationError: Missing API keys for fallback models
            FallbackExhaustedError: All fallback models failed

        Example:
            >>> import asyncio
            >>> client = Client()
            >>> result = await client.aembed(
            ...     model="voyage-3",
            ...     input="hello world",
            ...     fallbacks=["text-embedding-3-small"]
            ... )

        """
        # Normalize input to list
        inputs = [input] if isinstance(input, str) else input

        # Determine effective fallbacks (per-request overrides client-level)
        effective_fallbacks = fallbacks if fallbacks is not None else self.fallbacks
        effective_allow_unsafe = (
            allow_unsafe_fallback
            if allow_unsafe_fallback is not None
            else self.allow_unsafe_fallback
        )

        # If fallbacks are configured, use fallback logic
        if effective_fallbacks is not None:
            # Construct full model string with provider if specified
            full_model = f"{provider}:{model}" if provider and ":" not in model else model

            fallback_configs = self._normalize_fallbacks(effective_fallbacks)
            return await self._aembed_with_fallbacks(
                primary_model=full_model,
                inputs=inputs,
                fallback_configs=fallback_configs,
                allow_unsafe=effective_allow_unsafe,
                input_type=input_type,
                dimensions=dimensions,
                api_key=api_key,
                **kwargs,
            )

        # No fallbacks - use original logic
        # Parse provider and model
        provider_name, model_name = self._parse_model_string(model, provider)

        if self.verbose:
            print(f"Using provider: {provider_name}, model: {model_name}")

        # Get model info for feature validation
        model_info = self._catalog.get_model_info(provider_name, model_name)

        # Validate dimensions support if dimensions is provided
        if dimensions is not None:
            if not model_info.supports_dimensions:
                raise UnsupportedFeatureError(
                    f"Model '{model_name}' does not support custom dimensions",
                    model=model_name,
                    provider=provider_name,
                    feature="dimensions",
                )

        # Validate input_type support if input_type is provided
        if input_type is not None:
            if not model_info.supports_input_type:
                raise UnsupportedFeatureError(
                    f"Model '{model_name}' does not support input_type parameter",
                    model=model_name,
                    provider=provider_name,
                    feature="input_type",
                )

        # Get provider instance
        provider_instance = self._get_provider(provider_name)

        # Call provider's aembed method
        return await provider_instance.aembed(
            model=model_name,
            inputs=inputs,
            input_type=input_type,
            dimensions=dimensions,
            api_key=api_key,
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

    def tokenize(
        self,
        model: str,
        input: Union[str, List[str]],
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> TokenizeResponse:
        """Tokenize input text(s) without generating embeddings.

        Useful for counting tokens before making embedding requests.

        Args:
            model: Model name or "provider:model" string
            input: Single text string or list of text strings
            provider: Optional provider name (if not in model string)
            **kwargs: Additional provider-specific parameters

        Returns:
            TokenizeResponse with token counts

        Example:
            >>> client = Client()
            >>> result = client.tokenize(
            ...     model="voyage-3",
            ...     input="hello world",
            ...     provider="voyageai"
            ... )
            >>> print(result.token_count)  # 2

        """
        # Parse provider and model
        provider_name, model_name = self._parse_model_string(model, provider)

        if self.verbose:
            print(f"Tokenizing with provider: {provider_name}, model: {model_name}")

        # Get provider instance
        provider_instance = self._get_provider(provider_name)

        # Normalize input to list
        inputs = [input] if isinstance(input, str) else input

        # Call provider's tokenize method
        return provider_instance.tokenize(
            model=model_name,
            inputs=inputs,
            **kwargs,
        )

    def get_compatible_fallbacks(self, model: str) -> List[str]:
        """Get models with dimensions compatible with the given model.

        Returns models that can produce embeddings with the same dimensions
        as the specified model, making them suitable fallback candidates.

        Args:
            model: Model name or "provider:model" string

        Returns:
            List of compatible model names (in "provider:model" format)

        Example:
            >>> client = Client()
            >>> fallbacks = client.get_compatible_fallbacks("voyage-3")
            >>> print(fallbacks)
            ['voyageai:voyage-3-lite', 'cohere:embed-english-v3.0', ...]

        """
        # Get target model's dimensions
        provider_name, model_name = self._parse_model_string(model)
        model_info = self._catalog.get_model_info(provider_name, model_name)
        target_dims = model_info.dimensions

        compatible = []
        all_models = self._catalog.list_models()

        for m in all_models:
            # Skip the same model
            if m.provider == provider_name and m.name == model_name:
                continue

            # Check if model can produce target dimensions
            if m.supports_dimensions or m.dimensions == target_dims:
                compatible.append(f"{m.provider}:{m.name}")

        return compatible

    def close(self) -> None:
        """Close sync HTTP client."""
        self._http_client.close()

    async def aclose(self) -> None:
        """Close async HTTP client."""
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
