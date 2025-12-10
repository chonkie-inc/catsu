"""OpenAI embedding provider implementation.

Provides integration with OpenAI's embedding API, supporting text-embedding-3-large,
text-embedding-3-small, and text-embedding-ada-002 models with retry logic,
cost tracking, and local tokenization via tiktoken.
"""

import time
from typing import Any, Dict, List, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..catalog import ModelCatalog
from ..models import EmbedResponse, TokenizeResponse, Usage
from ..utils.errors import InvalidInputError, NetworkError, ProviderError, TimeoutError
from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI embedding provider.

    Implements the OpenAI embeddings API with support for text-embedding-3-large,
    text-embedding-3-small, and text-embedding-ada-002 models.

    Features:
    - Sync and async embedding generation
    - Local tokenization via tiktoken
    - Automatic retry with exponential backoff
    - Cost and latency tracking
    - Configurable output dimensions (for text-embedding-3 models)

    API Documentation: https://platform.openai.com/docs/api-reference/embeddings

    Example:
        >>> provider = OpenAIProvider(
        ...     http_client=httpx.Client(),
        ...     async_http_client=httpx.AsyncClient(),
        ...     api_key="your-api-key",
        ... )
        >>> response = provider.embed(
        ...     model="text-embedding-3-small",
        ...     inputs=["hello world"],
        ... )

    """

    API_BASE_URL = "https://api.openai.com/v1"
    PROVIDER_NAME = "openai"

    def __init__(
        self,
        http_client: httpx.Client,
        async_http_client: httpx.AsyncClient,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        verbose: bool = False,
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            http_client: Synchronous HTTP client
            async_http_client: Asynchronous HTTP client
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            max_retries: Maximum retry attempts (default: 3)
            verbose: Enable verbose logging (default: False)

        """
        super().__init__(
            http_client=http_client,
            async_http_client=async_http_client,
            api_key=api_key,
            max_retries=max_retries,
            verbose=verbose,
        )
        self._tokenizers: Dict[str, Any] = {}  # Cache for loaded tokenizers

    def _get_tokenizer(self, model: str) -> Any:
        """Get or load tokenizer for a model.

        Args:
            model: Model name

        Returns:
            Tokenizer wrapper instance (tiktoken)

        Raises:
            ImportError: If tiktoken library not installed
            ProviderError: If tokenizer cannot be loaded

        """
        from catsu.utils import load_tokenizer

        # Check cache first
        if model in self._tokenizers:
            return self._tokenizers[model]

        # Get model info to find tokenizer config
        catalog = ModelCatalog()

        try:
            model_info = catalog.get_model_info(self.PROVIDER_NAME, model)
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

    def _build_request_payload(
        self,
        model: str,
        inputs: List[str],
        dimensions: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build API request payload.

        Args:
            model: Model name
            inputs: List of input texts
            dimensions: Optional output dimensions (for text-embedding-3 models)
            **kwargs: Additional parameters

        Returns:
            Request payload dictionary

        """
        payload: Dict[str, Any] = {
            "input": inputs,
            "model": model,
        }

        # Add dimensions if provided (only for text-embedding-3 models)
        if dimensions is not None:
            if not model.startswith("text-embedding-3"):
                raise InvalidInputError(
                    "dimensions parameter is only supported for text-embedding-3 models",
                    parameter="dimensions",
                )
            payload["dimensions"] = dimensions

        # Add any additional parameters (excluding input_type which OpenAI doesn't support)
        for key, value in kwargs.items():
            if key != "input_type":
                payload[key] = value

        return payload

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
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
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
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
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

    def _parse_response(
        self,
        response_data: Dict[str, Any],
        model: str,
        input_count: int,
        latency_ms: float,
        cost_per_million: float,
    ) -> EmbedResponse:
        """Parse API response into EmbedResponse.

        Args:
            response_data: API response JSON
            model: Model name
            input_count: Number of inputs
            latency_ms: Request latency
            cost_per_million: Cost per million tokens

        Returns:
            EmbedResponse object

        """
        # Extract embeddings
        embeddings = []
        for item in response_data.get("data", []):
            embeddings.append(item["embedding"])

        # Get dimensions from first embedding
        dimensions = len(embeddings[0]) if embeddings else 0

        # Extract usage information
        usage_data = response_data.get("usage", {})
        total_tokens = usage_data.get("total_tokens", 0)

        # Calculate cost
        cost = self._calculate_cost(total_tokens, cost_per_million)

        # Create Usage object
        usage = Usage(
            tokens=total_tokens,
            cost=cost,
        )

        return EmbedResponse(
            embeddings=embeddings,
            model=model,
            provider=self.PROVIDER_NAME,
            dimensions=dimensions,
            usage=usage,
            latency_ms=latency_ms,
            input_count=input_count,
            input_type="document",  # OpenAI doesn't have input_type
        )

    def embed(
        self,
        model: str,
        inputs: List[str],
        input_type: Optional[str] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings using OpenAI API (synchronous).

        Args:
            model: Model name (e.g., "text-embedding-3-small", "text-embedding-3-large")
            inputs: List of input texts
            input_type: Ignored (OpenAI doesn't support input_type)
            dimensions: Optional output dimensions (only for text-embedding-3 models)
            api_key: Optional API key override for this request
            **kwargs: Additional API parameters

        Returns:
            EmbedResponse with embeddings, usage, and metadata

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit is exceeded
            InvalidInputError: If inputs are invalid
            ProviderError: For other API errors

        Example:
            >>> response = provider.embed(
            ...     model="text-embedding-3-small",
            ...     inputs=["hello", "world"],
            ... )
            >>> print(response.embeddings[0][:5])
            [0.1, 0.2, 0.3, 0.4, 0.5]

        """
        # Validate inputs and dimensions using Pydantic
        params = self._validate_inputs(inputs, dimensions=dimensions)

        # Get model info for cost calculation
        catalog = ModelCatalog()
        model_info = catalog.get_model_info(self.PROVIDER_NAME, model)

        # Build request
        url = f"{self.API_BASE_URL}/embeddings"
        payload = self._build_request_payload(model, params.inputs, params.dimensions, **kwargs)
        headers = self._get_headers(api_key)

        # Make request with retry logic
        start_time = time.time()
        response = self._make_request_with_retry(url, payload, headers)
        latency_ms = self._measure_latency(start_time)

        # Parse response
        response_data = response.json()

        return self._parse_response(
            response_data=response_data,
            model=model,
            input_count=len(params.inputs),
            latency_ms=latency_ms,
            cost_per_million=model_info.cost_per_million_tokens,
        )

    async def aembed(
        self,
        model: str,
        inputs: List[str],
        input_type: Optional[str] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings using OpenAI API (asynchronous).

        Args:
            model: Model name (e.g., "text-embedding-3-small", "text-embedding-3-large")
            inputs: List of input texts
            input_type: Ignored (OpenAI doesn't support input_type)
            dimensions: Optional output dimensions (only for text-embedding-3 models)
            api_key: Optional API key override for this request
            **kwargs: Additional API parameters

        Returns:
            EmbedResponse with embeddings, usage, and metadata

        Example:
            >>> response = await provider.aembed(
            ...     model="text-embedding-3-small",
            ...     inputs=["hello", "world"],
            ... )

        """
        # Validate inputs and dimensions using Pydantic
        params = self._validate_inputs(inputs, dimensions=dimensions)

        # Get model info for cost calculation
        catalog = ModelCatalog()
        model_info = catalog.get_model_info(self.PROVIDER_NAME, model)

        # Build request
        url = f"{self.API_BASE_URL}/embeddings"
        payload = self._build_request_payload(model, params.inputs, params.dimensions, **kwargs)
        headers = self._get_headers(api_key)

        # Make async request with retry logic
        start_time = time.time()
        response = await self._make_request_with_retry_async(url, payload, headers)
        latency_ms = self._measure_latency(start_time)

        # Parse response
        response_data = response.json()

        return self._parse_response(
            response_data=response_data,
            model=model,
            input_count=len(params.inputs),
            latency_ms=latency_ms,
            cost_per_million=model_info.cost_per_million_tokens,
        )

    def tokenize(
        self,
        model: str,
        inputs: List[str],
        **kwargs: Any,
    ) -> TokenizeResponse:
        """Tokenize inputs using local tiktoken tokenizer.

        Loads the tiktoken encoding and counts tokens locally
        without making an API call.

        Args:
            model: Model name
            inputs: List of input texts
            **kwargs: Additional parameters (ignored)

        Returns:
            TokenizeResponse with token count

        Raises:
            ImportError: If tiktoken library not installed
            ProviderError: If tokenizer cannot be loaded

        Example:
            >>> response = provider.tokenize(
            ...     model="text-embedding-3-small",
            ...     inputs=["hello world", "foo bar"]
            ... )
            >>> print(response.token_count)
            4

        """
        # Validate inputs
        params = self._validate_inputs(inputs)

        # Get tokenizer
        tokenizer = self._get_tokenizer(model)

        # Tokenize all inputs and count tokens
        total_tokens = 0
        for text in params.inputs:
            total_tokens += tokenizer.count_tokens(text)

        return TokenizeResponse(
            tokens=None,  # Don't return actual token IDs
            token_count=total_tokens,
            model=model,
            provider=self.PROVIDER_NAME,
        )
