"""Cohere embedding provider implementation.

Provides integration with Cohere's embedding API, supporting all Cohere embedding models
with retry logic, cost tracking, and local tokenization via HuggingFace.
"""

import time
from typing import Any, Dict, List, Optional

from ..models import EmbedResponse, TokenizeResponse, Usage
from ..utils.errors import InvalidInputError
from .base import BaseProvider


class CohereProvider(BaseProvider):
    """Cohere embedding provider.

    Implements the Cohere embeddings API with support for all Cohere embedding models,
    including embed-english-v3.0, embed-multilingual-v3.0, and their light variants.

    Features:
    - Sync and async embedding generation
    - Local tokenization (HuggingFace tokenizers)
    - Automatic retry with exponential backoff
    - Cost and latency tracking
    - Input type specification (search_document, search_query, classification, clustering)
    - Truncation options

    API Documentation: https://docs.cohere.com/reference/embed

    """

    API_BASE_URL = "https://api.cohere.com/v1"
    PROVIDER_NAME = "cohere"

    def _build_request_payload(
        self,
        model: str,
        inputs: List[str],
        input_type: Optional[str] = None,
        truncate: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build API request payload."""
        payload: Dict[str, Any] = {
            "texts": inputs,  # Cohere uses "texts" instead of "input"
            "model": model,
            "embedding_types": ["float"],
        }

        # Map standard input_type to Cohere's API values
        if input_type:
            if input_type not in ("query", "document"):
                raise InvalidInputError(
                    f"input_type must be 'query' or 'document', got '{input_type}'",
                    parameter="input_type",
                )
            cohere_input_type = (
                "search_query" if input_type == "query" else "search_document"
            )
            payload["input_type"] = cohere_input_type

        if truncate:
            valid_truncate = ("NONE", "START", "END")
            if truncate not in valid_truncate:
                raise InvalidInputError(
                    f"truncate must be one of {valid_truncate}, got '{truncate}'",
                    parameter="truncate",
                )
            payload["truncate"] = truncate

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
        # Cohere returns embeddings in embeddings.float
        embeddings = response_data.get("embeddings", {}).get("float", [])
        dimensions = len(embeddings[0]) if embeddings else 0

        # Cohere provides token count in meta.billed_units
        meta = response_data.get("meta", {})
        billed_units = meta.get("billed_units", {})
        total_tokens = billed_units.get("input_tokens", 0)
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
        truncate: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings using Cohere API (synchronous).

        Args:
            model: Model name (e.g., "embed-english-v3.0", "embed-multilingual-v3.0")
            inputs: List of input texts
            input_type: Input type ("query" or "document")
            truncate: Truncation option ("NONE", "START", "END")
            api_key: Optional API key override for this request
            **kwargs: Additional API parameters

        Returns:
            EmbedResponse with embeddings, usage, and metadata

        """
        params = self._validate_inputs(inputs, input_type=input_type)
        model_info = self.catalog.get_model_info(self.PROVIDER_NAME, model)

        url = f"{self.API_BASE_URL}/embed"
        payload = self._build_request_payload(
            model, params.inputs, params.input_type, truncate, **kwargs
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
        truncate: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings using Cohere API (asynchronous).

        Args:
            model: Model name (e.g., "embed-english-v3.0", "embed-multilingual-v3.0")
            inputs: List of input texts
            input_type: Input type ("query" or "document")
            truncate: Truncation option ("NONE", "START", "END")
            api_key: Optional API key override for this request
            **kwargs: Additional API parameters

        Returns:
            EmbedResponse with embeddings, usage, and metadata

        """
        params = self._validate_inputs(inputs, input_type=input_type)
        model_info = self.catalog.get_model_info(self.PROVIDER_NAME, model)

        url = f"{self.API_BASE_URL}/embed"
        payload = self._build_request_payload(
            model, params.inputs, params.input_type, truncate, **kwargs
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
