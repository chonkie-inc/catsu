"""Mistral AI embedding provider implementation.

Provides integration with Mistral AI's embedding API, supporting mistral-embed
and codestral-embed models with retry logic, cost tracking, and local tokenization.
"""

import time
from typing import Any, Dict, List, Optional

from ..models import EmbedResponse, TokenizeResponse, Usage
from .base import BaseProvider


class MistralProvider(BaseProvider):
    """Mistral AI embedding provider.

    Implements the Mistral AI embeddings API with support for mistral-embed
    and codestral-embed models.

    Features:
    - Sync and async embedding generation
    - Local tokenization (tiktoken)
    - Automatic retry with exponential backoff
    - Cost and latency tracking
    - Dimension configuration (for codestral-embed)
    - Batch embeddings (up to 512 texts)

    API Documentation: https://docs.mistral.ai/capabilities/embeddings

    """

    API_BASE_URL = "https://api.mistral.ai/v1"
    PROVIDER_NAME = "mistral"

    def _build_request_payload(
        self,
        model: str,
        inputs: List[str],
        encoding_format: Optional[str] = None,
        dimensions: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build API request payload."""
        payload: Dict[str, Any] = {
            "input": inputs,
            "model": model,
        }

        if encoding_format:
            payload["encoding_format"] = encoding_format

        # Map to Mistral's "output_dimension" parameter
        if dimensions is not None:
            payload["output_dimension"] = dimensions

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
        encoding_format: Optional[str] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings using Mistral AI API (synchronous).

        Args:
            model: Model name (e.g., "mistral-embed", "codestral-embed-2505")
            inputs: List of input texts (up to 512 texts for batch)
            input_type: Input type ("query" or "document")
            encoding_format: Output encoding format (e.g., "float", "int8")
            dimensions: Output dimensions (up to 3072 for codestral-embed-2505)
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
            encoding_format=encoding_format,
            dimensions=params.dimensions,
            **kwargs,
        )
        headers = self._get_headers(api_key)

        start_time = time.time()
        response = self._make_request_with_retry(url, payload, headers)
        latency_ms = self._measure_latency(start_time)

        # Note: Mistral API doesn't use input_type, but we validate it for consistency
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
        encoding_format: Optional[str] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings using Mistral AI API (asynchronous).

        Args:
            model: Model name (e.g., "mistral-embed", "codestral-embed-2505")
            inputs: List of input texts (up to 512 texts for batch)
            input_type: Input type ("query" or "document")
            encoding_format: Output encoding format (e.g., "float", "int8")
            dimensions: Output dimensions (up to 3072 for codestral-embed-2505)
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
            encoding_format=encoding_format,
            dimensions=params.dimensions,
            **kwargs,
        )
        headers = self._get_headers(api_key)

        start_time = time.time()
        response = await self._make_request_with_retry_async(url, payload, headers)
        latency_ms = self._measure_latency(start_time)

        # Note: Mistral API doesn't use input_type, but we validate it for consistency
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
