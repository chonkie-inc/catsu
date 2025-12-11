"""Mixedbread AI embedding provider implementation.

Provides integration with Mixedbread AI's embedding API, supporting mxbai-embed
models with Matryoshka Representation Learning, quantization support, retry logic,
cost tracking, and local tokenization.
"""

from typing import Any, Dict, List, Literal, Optional

from ..models import EmbedResponse, TokenizeResponse, Usage
from ..utils import Timer
from .base import BaseProvider


class MixedbreadProvider(BaseProvider):
    """Mixedbread AI embedding provider.

    Implements the Mixedbread AI embeddings API with support for mxbai-embed models
    featuring Matryoshka Representation Learning for flexible dimensions.

    Features:
    - Sync and async embedding generation
    - Local tokenization (HuggingFace tokenizers)
    - Automatic retry with exponential backoff
    - Cost and latency tracking
    - Configurable dimensions via Matryoshka Representation Learning
    - Input type specification (query vs document with prompt injection)
    - Normalized embeddings (default: true)
    - Multiple encoding formats (float, binary, int8, etc.)

    API Documentation: https://www.mixedbread.com/api-reference

    """

    API_BASE_URL = "https://api.mixedbread.com/v1"
    PROVIDER_NAME = "mixedbread"

    def _map_input_type_to_prompt(self, input_type: Optional[str]) -> Optional[str]:
        """Map generic input_type to Mixedbread prompt."""
        if input_type == "query":
            return "Represent this sentence for searching relevant passages: "
        return None

    def _build_request_payload(
        self,
        model: str,
        inputs: List[str],
        input_type: Optional[str] = None,
        dimensions: Optional[int] = None,
        normalized: bool = True,
        encoding_format: str = "float",
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build API request payload."""
        payload: Dict[str, Any] = {
            "input": inputs,
            "model": model,
            "normalized": normalized,
            "encoding_format": encoding_format,
        }

        # Add dimensions if provided (validation handled by model catalog)
        if dimensions is not None:
            payload["dimensions"] = dimensions

        # Add prompt - use provided prompt or map from input_type
        if prompt is not None:
            payload["prompt"] = prompt
        elif input_type is not None:
            mapped_prompt = self._map_input_type_to_prompt(input_type)
            if mapped_prompt:
                payload["prompt"] = mapped_prompt

        # Add any additional parameters
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
        dimensions = response_data.get("dimensions", len(embeddings[0]) if embeddings else 0)

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
        input_type: Optional[Literal["query", "document"]] = None,
        dimensions: Optional[int] = None,
        normalized: bool = True,
        encoding_format: str = "float",
        prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings using Mixedbread AI API (synchronous).

        Args:
            model: Model name (e.g., "mxbai-embed-large-v1")
            inputs: List of input texts
            input_type: Input type ("query" or "document") - maps to prompt injection
            dimensions: Optional output dimensions (Matryoshka truncation)
            normalized: Apply normalization to embeddings (default: true)
            encoding_format: Output format ("float", "binary", "int8", etc.)
            prompt: Custom prompt to prepend to inputs (overrides input_type mapping)
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
            input_type=params.input_type,
            dimensions=params.dimensions,
            normalized=normalized,
            encoding_format=encoding_format,
            prompt=prompt,
            **kwargs,
        )
        headers = self._get_headers(api_key)

        with Timer() as timer:
            response = self._make_request_with_retry(url, payload, headers)

        return self._parse_response(
            response_data=response.json(),
            model=model,
            input_count=len(params.inputs),
            input_type=params.input_type or "document",
            latency_ms=timer.elapsed_ms,
            cost_per_million=model_info.cost_per_million_tokens,
        )

    async def aembed(
        self,
        model: str,
        inputs: List[str],
        input_type: Optional[Literal["query", "document"]] = None,
        dimensions: Optional[int] = None,
        normalized: bool = True,
        encoding_format: str = "float",
        prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings using Mixedbread AI API (asynchronous).

        Args:
            model: Model name (e.g., "mxbai-embed-large-v1")
            inputs: List of input texts
            input_type: Input type ("query" or "document") - maps to prompt injection
            dimensions: Optional output dimensions (Matryoshka truncation)
            normalized: Apply normalization to embeddings (default: true)
            encoding_format: Output format ("float", "binary", "int8", etc.)
            prompt: Custom prompt to prepend to inputs (overrides input_type mapping)
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
            input_type=params.input_type,
            dimensions=params.dimensions,
            normalized=normalized,
            encoding_format=encoding_format,
            prompt=prompt,
            **kwargs,
        )
        headers = self._get_headers(api_key)

        with Timer() as timer:
            response = await self._make_request_with_retry_async(url, payload, headers)

        return self._parse_response(
            response_data=response.json(),
            model=model,
            input_count=len(params.inputs),
            input_type=params.input_type or "document",
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
