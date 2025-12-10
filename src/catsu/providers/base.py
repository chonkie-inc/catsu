"""Base provider class for embedding providers.

Defines the abstract interface that all embedding providers must implement.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import httpx
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..models import EmbedParams, EmbedResponse, TokenizeResponse
from ..utils.errors import (
    AuthenticationError,
    InvalidInputError,
    NetworkError,
    ProviderError,
    RateLimitError,
    TimeoutError,
)

if TYPE_CHECKING:
    from ..catalog import ModelCatalog


class BaseProvider(ABC):
    """Abstract base class for embedding providers.

    All provider implementations must inherit from this class and implement
    the abstract methods for embedding and tokenization.

    Attributes:
        http_client: Synchronous HTTP client for API requests
        async_http_client: Asynchronous HTTP client for API requests
        catalog: Model catalog for metadata lookup
        api_key: API key for authentication
        max_retries: Maximum number of retry attempts
        verbose: Enable verbose logging

    """

    # Subclasses must define these
    PROVIDER_NAME: str = ""
    API_BASE_URL: str = ""

    def __init__(
        self,
        http_client: httpx.Client,
        async_http_client: httpx.AsyncClient,
        catalog: "ModelCatalog",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        verbose: bool = False,
    ) -> None:
        """Initialize the base provider.

        Args:
            http_client: Synchronous HTTP client
            async_http_client: Asynchronous HTTP client
            catalog: Model catalog for metadata lookup
            api_key: API key for authentication
            max_retries: Maximum retry attempts (default: 3)
            verbose: Enable verbose logging (default: False)

        """
        self.http_client = http_client
        self.async_http_client = async_http_client
        self.catalog = catalog
        self.api_key = api_key
        self.max_retries = max_retries
        self.verbose = verbose
        self._tokenizers: Dict[str, Any] = {}  # Cache for loaded tokenizers

    @abstractmethod
    def embed(
        self,
        model: str,
        inputs: List[str],
        input_type: Optional[Literal["query", "document"]] = None,
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
        input_type: Optional[Literal["query", "document"]] = None,
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

    def _validate_inputs(
        self,
        inputs: List[str],
        input_type: Optional[Literal["query", "document"]] = None,
        dimensions: Optional[int] = None,
    ) -> "EmbedParams":
        """Validate input parameters using Pydantic.

        Validates inputs, input_type, and dimensions using the EmbedParams model.
        Provides centralized, type-safe validation.

        Args:
            inputs: List of input texts
            input_type: Optional input type ("query" or "document")
            dimensions: Optional output dimensions

        Returns:
            Validated EmbedParams object

        Raises:
            InvalidInputError: If any parameter is invalid

        """
        try:
            return EmbedParams(
                inputs=inputs,
                input_type=input_type,
                dimensions=dimensions,
            )
        except ValidationError as e:
            # Extract the first error for a cleaner message
            error = e.errors()[0]
            field = error["loc"][0] if error["loc"] else "parameter"
            message = error["msg"]
            raise InvalidInputError(message, parameter=str(field))

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

    def _get_tokenizer(self, model: str) -> Any:
        """Get or load tokenizer for a model.

        Args:
            model: Model name

        Returns:
            Tokenizer wrapper instance (HuggingFace or tiktoken)

        Raises:
            ImportError: If required tokenizer library not installed
            ProviderError: If tokenizer cannot be loaded

        """
        from catsu.utils import load_tokenizer

        # Check cache first
        if model in self._tokenizers:
            return self._tokenizers[model]

        try:
            model_info = self.catalog.get_model_info(self.PROVIDER_NAME, model)
        except Exception as e:
            raise ProviderError(
                message=f"Could not find model info for '{model}'",
                provider=self.PROVIDER_NAME,
            ) from e

        if not model_info.tokenizer:
            raise ProviderError(
                message=f"No tokenizer configured for model '{model}'",
                provider=self.PROVIDER_NAME,
            )

        # Load tokenizer using unified utility
        try:
            self._log(f"Loading tokenizer for {model}")
            tokenizer = load_tokenizer(model_info.tokenizer)
            self._tokenizers[model] = tokenizer
            return tokenizer
        except Exception as e:
            raise ProviderError(
                message=f"Failed to load tokenizer for '{model}': {str(e)}",
                provider=self.PROVIDER_NAME,
            ) from e

    def _get_headers(self, api_key: Optional[str] = None) -> Dict[str, str]:
        """Get HTTP headers for API requests.

        Default implementation uses Bearer token auth. Override in subclasses
        if the provider uses a different auth scheme.

        Args:
            api_key: Optional override API key (uses self.api_key if not provided)

        Returns:
            Dictionary of headers including authorization

        """
        effective_key = self._get_effective_api_key(api_key)
        return {
            "Authorization": f"Bearer {effective_key}",
            "Content-Type": "application/json",
        }

    def _make_request_with_retry(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
    ) -> httpx.Response:
        """Make HTTP request with exponential backoff retry logic.

        Uses tenacity for automatic retry with exponential backoff.

        Args:
            url: API endpoint URL
            payload: Request payload
            headers: Request headers

        Returns:
            HTTP response

        Raises:
            NetworkError: For network-related errors
            TimeoutError: For timeout errors
            ProviderError: For API errors

        """

        @retry(
            retry=retry_if_exception_type((
                httpx.TimeoutException,
                httpx.NetworkError,
                RateLimitError,
            )),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        def _do_request() -> httpx.Response:
            response = self.http_client.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                return response

            # Handle HTTP errors
            self._handle_http_error(response, self.PROVIDER_NAME)
            return response  # Won't reach here due to exception

        try:
            return _do_request()
        except httpx.TimeoutException as e:
            raise TimeoutError(
                message=f"Request timed out after {self.max_retries} attempts",
                provider=self.PROVIDER_NAME,
                timeout=self.http_client.timeout.read,
            ) from e
        except httpx.NetworkError as e:
            raise NetworkError(
                message=f"Network error: {str(e)}",
                provider=self.PROVIDER_NAME,
            ) from e

    async def _make_request_with_retry_async(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
    ) -> httpx.Response:
        """Make async HTTP request with exponential backoff retry logic.

        Uses tenacity for automatic retry with exponential backoff.

        Args:
            url: API endpoint URL
            payload: Request payload
            headers: Request headers

        Returns:
            HTTP response

        Raises:
            NetworkError: For network-related errors
            TimeoutError: For timeout errors
            ProviderError: For API errors

        """

        @retry(
            retry=retry_if_exception_type((
                httpx.TimeoutException,
                httpx.NetworkError,
                RateLimitError,
            )),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        async def _do_request() -> httpx.Response:
            response = await self.async_http_client.post(
                url, json=payload, headers=headers
            )

            if response.status_code == 200:
                return response

            # Handle HTTP errors
            self._handle_http_error(response, self.PROVIDER_NAME)
            return response  # Won't reach here due to exception

        try:
            return await _do_request()
        except httpx.TimeoutException as e:
            raise TimeoutError(
                message=f"Request timed out after {self.max_retries} attempts",
                provider=self.PROVIDER_NAME,
                timeout=self.async_http_client.timeout.read,
            ) from e
        except httpx.NetworkError as e:
            raise NetworkError(
                message=f"Network error: {str(e)}",
                provider=self.PROVIDER_NAME,
            ) from e
