"""Base provider class for embedding providers.

Defines the abstract interface that all embedding providers must implement.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import httpx

from ..models import EmbedResponse, TokenizeResponse
from ..utils.errors import (
    AuthenticationError,
    InvalidInputError,
    ProviderError,
    RateLimitError,
)


class BaseProvider(ABC):
    """Abstract base class for embedding providers.

    All provider implementations must inherit from this class and implement
    the abstract methods for embedding and tokenization.

    Attributes:
        http_client: Synchronous HTTP client for API requests
        async_http_client: Asynchronous HTTP client for API requests
        api_key: API key for authentication
        max_retries: Maximum number of retry attempts
        verbose: Enable verbose logging

    """

    def __init__(
        self,
        http_client: httpx.Client,
        async_http_client: httpx.AsyncClient,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        verbose: bool = False,
    ) -> None:
        """Initialize the base provider.

        Args:
            http_client: Synchronous HTTP client
            async_http_client: Asynchronous HTTP client
            api_key: API key for authentication
            max_retries: Maximum retry attempts (default: 3)
            verbose: Enable verbose logging (default: False)

        """
        self.http_client = http_client
        self.async_http_client = async_http_client
        self.api_key = api_key
        self.max_retries = max_retries
        self.verbose = verbose

    @abstractmethod
    def embed(
        self,
        model: str,
        inputs: List[str],
        input_type: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings for input texts (synchronous).

        Args:
            model: Model name
            inputs: List of input texts
            input_type: Optional input type hint ("query" or "document")
            **kwargs: Additional provider-specific parameters

        Returns:
            EmbedResponse with embeddings, usage, and metadata

        Raises:
            ProviderError: If API request fails
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit is exceeded
            InvalidInputError: If input is invalid

        """
        pass

    @abstractmethod
    async def aembed(
        self,
        model: str,
        inputs: List[str],
        input_type: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings for input texts (asynchronous).

        Args:
            model: Model name
            inputs: List of input texts
            input_type: Optional input type hint ("query" or "document")
            **kwargs: Additional provider-specific parameters

        Returns:
            EmbedResponse with embeddings, usage, and metadata

        Raises:
            ProviderError: If API request fails
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit is exceeded
            InvalidInputError: If input is invalid

        """
        pass

    @abstractmethod
    def tokenize(
        self,
        model: str,
        inputs: List[str],
        **kwargs: Any,
    ) -> TokenizeResponse:
        """Tokenize input texts without generating embeddings.

        Useful for counting tokens before making actual embedding requests.

        Args:
            model: Model name
            inputs: List of input texts
            **kwargs: Additional provider-specific parameters

        Returns:
            TokenizeResponse with token counts

        Raises:
            ProviderError: If API request fails
            NotImplementedError: If provider doesn't support tokenization

        """
        pass

    def _validate_inputs(self, inputs: List[str]) -> None:
        """Validate input texts.

        Args:
            inputs: List of input texts

        Raises:
            InvalidInputError: If inputs are invalid

        """
        if not inputs:
            raise InvalidInputError("inputs cannot be empty")

        if not isinstance(inputs, list):
            raise InvalidInputError(
                "inputs must be a list",
                parameter="inputs",
            )

        for i, text in enumerate(inputs):
            if not isinstance(text, str):
                raise InvalidInputError(
                    f"Input at index {i} must be a string, got {type(text).__name__}",
                    parameter=f"inputs[{i}]",
                )

            if not text.strip():
                raise InvalidInputError(
                    f"Input at index {i} cannot be empty or whitespace only",
                    parameter=f"inputs[{i}]",
                )

    def _validate_api_key(self, api_key: Optional[str] = None) -> None:
        """Validate that API key is present.

        Args:
            api_key: Optional override API key (uses self.api_key if not provided)

        Raises:
            AuthenticationError: If API key is missing

        """
        effective_key = api_key if api_key is not None else self.api_key
        if not effective_key:
            raise AuthenticationError(
                f"API key is required for {self.__class__.__name__}. "
                f"Set it via the Client or environment variable."
            )

    def _get_effective_api_key(self, api_key: Optional[str] = None) -> str:
        """Get the effective API key, validating it exists.

        Args:
            api_key: Optional override API key (uses self.api_key if not provided)

        Returns:
            The effective API key to use

        Raises:
            AuthenticationError: If no API key is available

        """
        effective_key = api_key if api_key is not None else self.api_key
        if not effective_key:
            raise AuthenticationError(
                f"API key is required for {self.__class__.__name__}. "
                f"Set it via the Client or environment variable."
            )
        return effective_key

    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled.

        Args:
            message: Message to log

        """
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")

    def _calculate_cost(self, tokens: int, cost_per_million: float) -> float:
        """Calculate cost for given token count.

        Args:
            tokens: Number of tokens
            cost_per_million: Cost per million tokens

        Returns:
            Total cost in USD

        """
        return (tokens / 1_000_000) * cost_per_million

    def _measure_latency(self, start_time: float) -> float:
        """Measure latency in milliseconds.

        Args:
            start_time: Start time from time.time()

        Returns:
            Latency in milliseconds

        """
        return (time.time() - start_time) * 1000

    def _handle_http_error(
        self,
        response: httpx.Response,
        provider_name: str,
    ) -> None:
        """Handle HTTP error responses.

        Args:
            response: HTTP response object
            provider_name: Name of the provider

        Raises:
            AuthenticationError: For 401 errors
            RateLimitError: For 429 errors
            ProviderError: For other errors

        """
        status_code = response.status_code

        # Try to get error message from response
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", str(error_data))
        except Exception:
            error_message = response.text or f"HTTP {status_code}"

        # Handle specific status codes
        if status_code == 401 or status_code == 403:
            raise AuthenticationError(
                message=f"Authentication failed: {error_message}",
                provider=provider_name,
            )
        elif status_code == 429:
            # Try to get retry_after header
            retry_after = response.headers.get("Retry-After")
            retry_after_seconds = int(retry_after) if retry_after else None

            raise RateLimitError(
                message=f"Rate limit exceeded: {error_message}",
                provider=provider_name,
                retry_after=retry_after_seconds,
            )
        else:
            raise ProviderError(
                message=f"API request failed: {error_message}",
                provider=provider_name,
                status_code=status_code,
            )

    def __repr__(self) -> str:
        """Return string representation of the provider."""
        return (
            f"{self.__class__.__name__}("
            f"max_retries={self.max_retries}, "
            f"verbose={self.verbose})"
        )
