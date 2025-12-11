"""Cloudflare Workers AI embedding provider implementation.

Provides integration with Cloudflare Workers AI embedding API, which hosts
open-source embedding models on Cloudflare's edge network. Supports BGE, Qwen3,
EmbeddingGemma, and PLaMo models with retry logic, cost tracking, and local tokenization.
"""

from typing import Any, Dict, List, Literal, Optional

from ..models import EmbedResponse, TokenizeResponse, Usage
from ..utils import Timer
from ..utils.errors import AuthenticationError
from .base import BaseProvider


class CloudflareProvider(BaseProvider):
    """Cloudflare Workers AI embedding provider.

    Implements the Cloudflare Workers AI embeddings API with support for
    BGE, Qwen3, EmbeddingGemma, and PLaMo embedding models running on the edge.

    Features:
    - Sync and async embedding generation
    - Local tokenization (HuggingFace tokenizers)
    - Automatic retry with exponential backoff
    - Cost and latency tracking
    - Edge-based inference for low latency
    - Pooling method configuration (mean or cls for BGE models)
    - OpenAI-compatible endpoint support

    API Documentation: https://developers.cloudflare.com/workers-ai/

    """

    API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts"
    PROVIDER_NAME = "cloudflare"

    def __init__(
        self,
        http_client,
        async_http_client,
        catalog,
        api_key: Optional[str] = None,
        account_id: Optional[str] = None,
        max_retries: int = 3,
        verbose: bool = False,
    ) -> None:
        """Initialize Cloudflare provider.

        Args:
            http_client: Synchronous HTTP client
            async_http_client: Asynchronous HTTP client
            catalog: Model catalog for metadata lookup
            api_key: Cloudflare API token
            account_id: Cloudflare account ID (required)
            max_retries: Maximum retry attempts
            verbose: Enable verbose logging

        """
        super().__init__(
            http_client, async_http_client, catalog, api_key, max_retries, verbose
        )
        self.account_id = account_id

    def _get_account_id(self, account_id: Optional[str] = None) -> str:
        """Get the effective account ID, validating it exists.

        Args:
            account_id: Optional override account ID

        Returns:
            The effective account ID to use

        Raises:
            AuthenticationError: If no account ID is available

        """
        effective_id = account_id if account_id is not None else self.account_id
        if not effective_id:
            raise AuthenticationError(
                f"Account ID is required for {self.__class__.__name__}. "
                f"Set it via the Client or environment variable."
            )
        return effective_id

    def _build_request_payload(
        self,
        model: str,
        inputs: List[str],
        pooling: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build API request payload."""
        payload: Dict[str, Any] = {
            "text": inputs,  # Cloudflare uses "text" instead of "input"
        }

        # Add pooling for BGE models
        if pooling is not None:
            payload["pooling"] = pooling

        # Add any additional parameters (excluding input_type and dimensions)
        for key, value in kwargs.items():
            if key not in ("input_type", "dimensions"):
                payload[key] = value

        return payload

    def _parse_response(
        self,
        response_data: Dict[str, Any],
        model: str,
        inputs: List[str],
        input_count: int,
        latency_ms: float,
        cost_per_million: float,
    ) -> EmbedResponse:
        """Parse API response into EmbedResponse."""
        embeddings = response_data.get("result", {}).get("data", [])
        shape = response_data.get("result", {}).get("shape", [])
        dimensions = (
            shape[1] if len(shape) > 1 else (len(embeddings[0]) if embeddings else 0)
        )

        # Cloudflare doesn't return token usage, calculate from inputs using tokenizer
        tokenizer = self._get_tokenizer(model)
        total_tokens = sum(tokenizer.count_tokens(text) for text in inputs)
        cost = self._calculate_cost(total_tokens, cost_per_million)

        return EmbedResponse(
            embeddings=embeddings,
            model=model,
            provider=self.PROVIDER_NAME,
            dimensions=dimensions,
            usage=Usage(tokens=total_tokens, cost=cost),
            latency_ms=latency_ms,
            input_count=input_count,
            input_type="document",
        )

    def embed(
        self,
        model: str,
        inputs: List[str],
        input_type: Optional[Literal["query", "document"]] = None,
        dimensions: Optional[int] = None,
        pooling: Optional[str] = None,
        api_key: Optional[str] = None,
        account_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings using Cloudflare Workers AI API (synchronous).

        Args:
            model: Model name (e.g., "@cf/baai/bge-base-en-v1.5")
            inputs: List of input texts
            input_type: Ignored (Cloudflare doesn't support input_type)
            dimensions: Ignored (Cloudflare doesn't support custom dimensions)
            pooling: Pooling method for BGE models ("mean" or "cls")
            api_key: Optional API key override for this request
            account_id: Optional account ID override for this request
            **kwargs: Additional API parameters

        Returns:
            EmbedResponse with embeddings, usage, and metadata

        """
        params = self._validate_inputs(
            inputs, input_type=input_type, dimensions=dimensions
        )
        model_info = self.catalog.get_model_info(self.PROVIDER_NAME, model)
        effective_account_id = self._get_account_id(account_id)

        url = f"{self.API_BASE_URL}/{effective_account_id}/ai/run/{model}"
        payload = self._build_request_payload(
            model, params.inputs, pooling=pooling, **kwargs
        )
        headers = self._get_headers(api_key)

        with Timer() as timer:
            response = self._make_request_with_retry(url, payload, headers)

        return self._parse_response(
            response_data=response.json(),
            model=model,
            inputs=params.inputs,
            input_count=len(params.inputs),
            latency_ms=timer.elapsed_ms,
            cost_per_million=model_info.cost_per_million_tokens,
        )

    async def aembed(
        self,
        model: str,
        inputs: List[str],
        input_type: Optional[Literal["query", "document"]] = None,
        dimensions: Optional[int] = None,
        pooling: Optional[str] = None,
        api_key: Optional[str] = None,
        account_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings using Cloudflare Workers AI API (asynchronous).

        Args:
            model: Model name (e.g., "@cf/baai/bge-base-en-v1.5")
            inputs: List of input texts
            input_type: Ignored (Cloudflare doesn't support input_type)
            dimensions: Ignored (Cloudflare doesn't support custom dimensions)
            pooling: Pooling method for BGE models ("mean" or "cls")
            api_key: Optional API key override for this request
            account_id: Optional account ID override for this request
            **kwargs: Additional API parameters

        Returns:
            EmbedResponse with embeddings, usage, and metadata

        """
        params = self._validate_inputs(
            inputs, input_type=input_type, dimensions=dimensions
        )
        model_info = self.catalog.get_model_info(self.PROVIDER_NAME, model)
        effective_account_id = self._get_account_id(account_id)

        url = f"{self.API_BASE_URL}/{effective_account_id}/ai/run/{model}"
        payload = self._build_request_payload(
            model, params.inputs, pooling=pooling, **kwargs
        )
        headers = self._get_headers(api_key)

        with Timer() as timer:
            response = await self._make_request_with_retry_async(url, payload, headers)

        return self._parse_response(
            response_data=response.json(),
            model=model,
            inputs=params.inputs,
            input_count=len(params.inputs),
            latency_ms=timer.elapsed_ms,
            cost_per_million=model_info.cost_per_million_tokens,
        )

    def tokenize(
        self,
        model: str,
        inputs: List[str],
        **kwargs: Any,
    ) -> TokenizeResponse:
        """Tokenize inputs using local tokenizer."""
        params = self._validate_inputs(inputs)
        tokenizer = self._get_tokenizer(model)
        total_tokens = sum(tokenizer.count_tokens(text) for text in params.inputs)

        return TokenizeResponse(
            tokens=None,
            token_count=total_tokens,
            model=model,
            provider=self.PROVIDER_NAME,
        )
