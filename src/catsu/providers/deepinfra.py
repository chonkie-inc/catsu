"""DeepInfra embedding provider implementation.

Provides integration with DeepInfra's embedding API, which hosts open-source
embedding models via an OpenAI-compatible API. Supports Qwen3-Embedding,
BGE, E5, GTE, and other popular models with retry logic, cost tracking,
and local tokenization.
"""

from typing import Any, Dict, List, Literal, Optional

from ..models import EmbedResponse, TokenizeResponse, Usage
from ..utils import Timer
from .base import BaseProvider


class DeepInfraProvider(BaseProvider):
    """DeepInfra embedding provider.

    Implements the DeepInfra embeddings API (OpenAI-compatible) with support for
    open-source embedding models including Qwen3-Embedding, BGE, E5, GTE,
    sentence-transformers, and more.

    Features:
    - Sync and async embedding generation
    - Local tokenization (HuggingFace tokenizers)
    - Automatic retry with exponential backoff
    - Cost and latency tracking
    - Wide range of open-source models

    API Documentation: https://deepinfra.com/models/embeddings/

    """

    API_BASE_URL = "https://api.deepinfra.com/v1/openai"
    PROVIDER_NAME = "deepinfra"

    def _build_request_payload(
        self,
        model: str,
        inputs: List[str],
        dimensions: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build API request payload."""
        payload: Dict[str, Any] = {
            "input": inputs,
            "model": model,
        }

        # Add dimensions if provided (validation handled by model catalog)
        if dimensions is not None:
            payload["dimensions"] = dimensions

        # Add any additional parameters (excluding input_type which DeepInfra doesn't support)
        for key, value in kwargs.items():
            if key != "input_type":
                payload[key] = value

        return payload

    def _parse_response(
        self,
        response_data: Dict[str, Any],
        model: str,
        input_count: int,
        latency_ms: float,
        cost_per_million: float,
    ) -> EmbedResponse:
        """Parse API response into EmbedResponse."""
        embeddings = [item["embedding"] for item in response_data.get("data", [])]
        dimensions = len(embeddings[0]) if embeddings else 0

        usage_data = response_data.get("usage", {})
        total_tokens = usage_data.get("total_tokens", 0)
        cost = self._calculate_cost(total_tokens, cost_per_million)

        return EmbedResponse(
            embeddings=embeddings,
            model=model,
            provider=self.PROVIDER_NAME,
            dimensions=dimensions,
            usage=Usage(tokens=total_tokens, cost=cost),
            latency_ms=latency_ms,
            input_count=input_count,
            input_type="document",  # DeepInfra doesn't have input_type
        )

    def embed(
        self,
        model: str,
        inputs: List[str],
        input_type: Optional[Literal["query", "document"]] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings using DeepInfra API (synchronous).

        Args:
            model: Model name (e.g., "BAAI/bge-base-en-v1.5", "Qwen/Qwen3-Embedding-8B")
            inputs: List of input texts
            input_type: Ignored (DeepInfra doesn't support input_type)
            dimensions: Optional output dimensions (only for Qwen3-Embedding models, 32-8192)
            api_key: Optional API key override for this request
            **kwargs: Additional API parameters

        Returns:
            EmbedResponse with embeddings, usage, and metadata

        """
        params = self._validate_inputs(
            inputs, input_type=input_type, dimensions=dimensions
        )
        model_info = self.catalog.get_model_info(self.PROVIDER_NAME, model)

        url = f"{self.API_BASE_URL}/embeddings"
        payload = self._build_request_payload(
            model, params.inputs, params.dimensions, **kwargs
        )
        headers = self._get_headers(api_key)

        with Timer() as timer:
            response = self._make_request_with_retry(url, payload, headers)

        return self._parse_response(
            response_data=response.json(),
            model=model,
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
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings using DeepInfra API (asynchronous).

        Args:
            model: Model name (e.g., "BAAI/bge-base-en-v1.5", "Qwen/Qwen3-Embedding-8B")
            inputs: List of input texts
            input_type: Ignored (DeepInfra doesn't support input_type)
            dimensions: Optional output dimensions (only for Qwen3-Embedding models, 32-8192)
            api_key: Optional API key override for this request
            **kwargs: Additional API parameters

        Returns:
            EmbedResponse with embeddings, usage, and metadata

        """
        params = self._validate_inputs(
            inputs, input_type=input_type, dimensions=dimensions
        )
        model_info = self.catalog.get_model_info(self.PROVIDER_NAME, model)

        url = f"{self.API_BASE_URL}/embeddings"
        payload = self._build_request_payload(
            model, params.inputs, params.dimensions, **kwargs
        )
        headers = self._get_headers(api_key)

        with Timer() as timer:
            response = await self._make_request_with_retry_async(url, payload, headers)

        return self._parse_response(
            response_data=response.json(),
            model=model,
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
