"""Jina AI embedding provider implementation.

Provides integration with Jina AI's embedding API, supporting all Jina embedding models
with retry logic, cost tracking, and local tokenization via HuggingFace.
"""

import time
from typing import Any, Dict, List, Optional

from ..models import EmbedResponse, TokenizeResponse, Usage
from .base import BaseProvider


class JinaAIProvider(BaseProvider):
    """Jina AI embedding provider.

    Implements the Jina AI embeddings API with support for all Jina embedding models,
    including jina-embeddings-v4, jina-embeddings-v3, v2 variants, and code models.

    Features:
    - Sync and async embedding generation
    - Local tokenization (HuggingFace tokenizers)
    - Automatic retry with exponential backoff
    - Cost and latency tracking
    - Task type specification
    - Dimension configuration (Matryoshka)
    - Normalized embeddings support

    API Documentation: https://jina.ai/embeddings/

    """

    API_BASE_URL = "https://api.jina.ai/v1"
    PROVIDER_NAME = "jinaai"

    def _map_input_type_to_task(self, input_type: Optional[str]) -> str:
        """Map generic input_type to Jina task."""
        if input_type == "query":
            return "retrieval.query"
        return "retrieval.passage"

    def _build_request_payload(
        self,
        model: str,
        inputs: List[str],
        task: Optional[str] = None,
        dimensions: Optional[int] = None,
        normalized: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build API request payload."""
        payload: Dict[str, Any] = {
            "input": inputs,
            "model": model,
            "normalized": normalized,
        }

        if task:
            payload["task"] = task

        if dimensions:
            payload["dimensions"] = dimensions

        payload.update(kwargs)
        return payload

    def _parse_response(
        self,
        response_data: Dict[str, Any],
        model: str,
        input_count: int,
        input_type: str,
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
            input_type=input_type,
        )

    def embed(
        self,
        model: str,
        inputs: List[str],
        input_type: Optional[str] = None,
        dimensions: Optional[int] = None,
        normalized: bool = True,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings using Jina AI API (synchronous).

        Args:
            model: Model name (e.g., "jina-embeddings-v3", "jina-embeddings-v4")
            inputs: List of input texts
            input_type: Input type ("query" or "document")
            dimensions: Output dimensions (for Matryoshka models)
            normalized: L2 normalize embeddings (default: True)
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
            model,
            params.inputs,
            task=self._map_input_type_to_task(params.input_type),
            dimensions=params.dimensions,
            normalized=normalized,
            **kwargs,
        )
        headers = self._get_headers(api_key)

        start_time = time.time()
        response = self._make_request_with_retry(url, payload, headers)
        latency_ms = self._measure_latency(start_time)

        return self._parse_response(
            response_data=response.json(),
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
        dimensions: Optional[int] = None,
        normalized: bool = True,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings using Jina AI API (asynchronous).

        Args:
            model: Model name (e.g., "jina-embeddings-v3", "jina-embeddings-v4")
            inputs: List of input texts
            input_type: Input type ("query" or "document")
            dimensions: Output dimensions (for Matryoshka models)
            normalized: L2 normalize embeddings (default: True)
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
            model,
            params.inputs,
            task=self._map_input_type_to_task(params.input_type),
            dimensions=params.dimensions,
            normalized=normalized,
            **kwargs,
        )
        headers = self._get_headers(api_key)

        start_time = time.time()
        response = await self._make_request_with_retry_async(url, payload, headers)
        latency_ms = self._measure_latency(start_time)

        return self._parse_response(
            response_data=response.json(),
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
