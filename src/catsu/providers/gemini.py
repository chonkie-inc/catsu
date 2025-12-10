"""Gemini API embedding provider implementation.

Provides integration with Google's Gemini API for embeddings, supporting gemini-embedding-001
and text-embedding models with retry logic, cost tracking, and local tokenization.
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


class GeminiProvider(BaseProvider):
    """Gemini API embedding provider.

    Implements the Gemini API embeddings with support for gemini-embedding-001
    and text-embedding models.

    Features:
    - Sync and async embedding generation
    - Local tokenization (tiktoken)
    - Automatic retry with exponential backoff
    - Cost and latency tracking
    - Task type specification
    - Flexible output dimensionality (Matryoshka)

    API Documentation: https://ai.google.dev/gemini-api/docs/embeddings

    Example:
        >>> provider = GeminiProvider(
        ...     http_client=httpx.Client(),
        ...     async_http_client=httpx.AsyncClient(),
        ...     api_key="your-api-key",
        ... )
        >>> response = provider.embed(
        ...     model="gemini-embedding-001",
        ...     inputs=["hello world"],
        ...     task_type="RETRIEVAL_QUERY"
        ... )

    """

    API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    PROVIDER_NAME = "gemini"

    def __init__(
        self,
        http_client: httpx.Client,
        async_http_client: httpx.AsyncClient,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        verbose: bool = False,
    ) -> None:
        """Initialize Gemini API provider.

        Args:
            http_client: Synchronous HTTP client
            async_http_client: Asynchronous HTTP client
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
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
            Tokenizer wrapper instance

        Raises:
            ImportError: If required tokenizer library not installed
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
            Dictionary of headers including API key

        """
        effective_key = self._get_effective_api_key(api_key)
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": effective_key,
        }

    def _build_request_payload(
        self,
        inputs: List[str],
        task_type: Optional[str] = None,
        dimensions: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build API request payload.

        Args:
            inputs: List of input texts
            task_type: Task type (e.g., "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT")
            dimensions: Output dimensions (128-3072)
            **kwargs: Additional parameters

        Returns:
            Request payload dictionary

        """
        # Convert inputs to Gemini's content format
        requests = []
        for text in inputs:
            request = {"content": {"parts": [{"text": text}]}}

            # Add task_type if provided
            if task_type:
                valid_types = (
                    "RETRIEVAL_QUERY",
                    "RETRIEVAL_DOCUMENT",
                    "SEMANTIC_SIMILARITY",
                    "CLASSIFICATION",
                    "CLUSTERING",
                    "QUESTION_ANSWERING",
                    "FACT_VERIFICATION",
                    "CODE_RETRIEVAL_QUERY",
                )
                if task_type not in valid_types:
                    raise InvalidInputError(
                        f"task_type must be one of {valid_types}, got '{task_type}'",
                        parameter="task_type",
                    )
                request["taskType"] = task_type

            # Add dimensions if provided
            # Map from standard "dimensions" to Gemini's API parameter "outputDimensionality"
            if dimensions:
                if dimensions < 128 or dimensions > 3072:
                    raise InvalidInputError(
                        f"dimensions must be between 128 and 3072, got {dimensions}",
                        parameter="dimensions",
                    )
                request["outputDimensionality"] = dimensions

            requests.append(request)

        payload: Dict[str, Any] = {"requests": requests}

        # Add any additional parameters
        payload.update(kwargs)

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
        input_type: str,
        latency_ms: float,
        cost_per_million: float,
    ) -> EmbedResponse:
        """Parse API response into EmbedResponse.

        Args:
            response_data: API response JSON
            model: Model name
            input_count: Number of inputs
            input_type: Input type
            latency_ms: Request latency
            cost_per_million: Cost per million tokens

        Returns:
            EmbedResponse object

        """
        # Extract embeddings from response
        embeddings = []
        for item in response_data.get("embeddings", []):
            embeddings.append(item["values"])

        # Get dimensions from first embedding
        dimensions = len(embeddings[0]) if embeddings else 0

        # Estimate token count (Gemini doesn't provide usage in response)
        # Use rough estimate: ~1 token per 4 characters
        total_chars = sum(len(text) for text in embeddings)
        estimated_tokens = max(1, total_chars // 4)

        # Calculate cost
        cost = self._calculate_cost(estimated_tokens, cost_per_million)

        # Create Usage object
        usage = Usage(
            tokens=estimated_tokens,
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
            input_type=input_type,
        )

    def embed(
        self,
        model: str,
        inputs: List[str],
        input_type: Optional[str] = None,
        task_type: Optional[str] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings using Gemini API (synchronous).

        Args:
            model: Model name (e.g., "gemini-embedding-001", "text-embedding-005")
            inputs: List of input texts
            input_type: Input type ("query" or "document")
            task_type: Gemini task type (e.g., "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT")
            dimensions: Output dimensions (128-3072, recommended: 768, 1536, 3072)
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
            ...     model="gemini-embedding-001",
            ...     inputs=["hello", "world"],
            ...     task_type="RETRIEVAL_QUERY"
            ... )
            >>> print(response.embeddings[0][:5])
            [0.1, 0.2, 0.3, 0.4, 0.5]

        """
        # Validate inputs, input_type, and dimensions using Pydantic
        params = self._validate_inputs(inputs, input_type=input_type, dimensions=dimensions)

        # Get model info for cost calculation
        catalog = ModelCatalog()
        model_info = catalog.get_model_info(self.PROVIDER_NAME, model)

        # Build request
        url = f"{self.API_BASE_URL}/models/{model}:batchEmbedContents"
        payload = self._build_request_payload(
            params.inputs,
            task_type=task_type,
            dimensions=params.dimensions,
            **kwargs,
        )
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
            input_type=params.input_type or "document",
            latency_ms=latency_ms,
            cost_per_million=model_info.cost_per_million_tokens,
        )

    async def aembed(
        self,
        model: str,
        inputs: List[str],
        input_type: Optional[str] = None,
        task_type: Optional[str] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings using Gemini API (asynchronous).

        Args:
            model: Model name (e.g., "gemini-embedding-001", "text-embedding-005")
            inputs: List of input texts
            input_type: Input type ("query" or "document")
            task_type: Gemini task type (e.g., "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT")
            dimensions: Output dimensions (128-3072)
            api_key: Optional API key override for this request
            **kwargs: Additional API parameters

        Returns:
            EmbedResponse with embeddings, usage, and metadata

        Example:
            >>> response = await provider.aembed(
            ...     model="gemini-embedding-001",
            ...     inputs=["hello", "world"],
            ...     task_type="RETRIEVAL_QUERY"
            ... )

        """
        # Validate inputs, input_type, and dimensions using Pydantic
        params = self._validate_inputs(inputs, input_type=input_type, dimensions=dimensions)

        # Get model info for cost calculation
        catalog = ModelCatalog()
        model_info = catalog.get_model_info(self.PROVIDER_NAME, model)

        # Build request
        url = f"{self.API_BASE_URL}/models/{model}:batchEmbedContents"
        payload = self._build_request_payload(
            params.inputs,
            task_type=task_type,
            dimensions=params.dimensions,
            **kwargs,
        )
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
            input_type=params.input_type or "document",
            latency_ms=latency_ms,
            cost_per_million=model_info.cost_per_million_tokens,
        )

    def tokenize(
        self,
        model: str,
        inputs: List[str],
        **kwargs: Any,
    ) -> TokenizeResponse:
        """Tokenize inputs using local tokenizer.

        Loads the appropriate tokenizer and counts tokens locally
        without making an API call.

        Args:
            model: Model name
            inputs: List of input texts
            **kwargs: Additional parameters (ignored)

        Returns:
            TokenizeResponse with token count

        Raises:
            ImportError: If required tokenizer library not installed
            ProviderError: If tokenizer cannot be loaded

        Example:
            >>> response = provider.tokenize(
            ...     model="gemini-embedding-001",
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
