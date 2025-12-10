"""Gemini API embedding provider implementation.

Provides integration with Google's Gemini API for embeddings, supporting gemini-embedding-001
and text-embedding models with retry logic, cost tracking, and local tokenization.
"""

import time
from typing import Any, Dict, List, Literal, Optional

from ..models import EmbedResponse, TokenizeResponse, Usage
from ..utils.errors import InvalidInputError
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

    """

    API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    PROVIDER_NAME = "gemini"

    def _get_headers(self, api_key: Optional[str] = None) -> Dict[str, str]:
        """Get HTTP headers for API requests (Gemini uses x-goog-api-key)."""
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
        """Build API request payload."""
        requests = []
        for text in inputs:
            request: Dict[str, Any] = {"content": {"parts": [{"text": text}]}}

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

            # Map to Gemini's "outputDimensionality" parameter
            if dimensions:
                if dimensions < 128 or dimensions > 3072:
                    raise InvalidInputError(
                        f"dimensions must be between 128 and 3072, got {dimensions}",
                        parameter="dimensions",
                    )
                request["outputDimensionality"] = dimensions

            requests.append(request)

        payload: Dict[str, Any] = {"requests": requests}
        payload.update(kwargs)
        return payload

    def _parse_response(
        self,
        response_data: Dict[str, Any],
        model: str,
        inputs: List[str],
        input_type: str,
        latency_ms: float,
        cost_per_million: float,
    ) -> EmbedResponse:
        """Parse API response into EmbedResponse."""
        embeddings = [item["values"] for item in response_data.get("embeddings", [])]
        dimensions = len(embeddings[0]) if embeddings else 0

        # Estimate token count from inputs (Gemini doesn't provide usage in response)
        # ~1 token per 4 characters
        total_chars = sum(len(text) for text in inputs)
        estimated_tokens = max(1, total_chars // 4)
        cost = self._calculate_cost(estimated_tokens, cost_per_million)

        return EmbedResponse(
            embeddings=embeddings,
            model=model,
            provider=self.PROVIDER_NAME,
            dimensions=dimensions,
            usage=Usage(tokens=estimated_tokens, cost=cost),
            latency_ms=latency_ms,
            input_count=len(inputs),
            input_type=input_type,
        )

    def embed(
        self,
        model: str,
        inputs: List[str],
        input_type: Optional[Literal["query", "document"]] = None,
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
            dimensions: Output dimensions (128-3072)
            api_key: Optional API key override for this request
            **kwargs: Additional API parameters

        Returns:
            EmbedResponse with embeddings, usage, and metadata

        """
        params = self._validate_inputs(
            inputs, input_type=input_type, dimensions=dimensions
        )
        model_info = self.catalog.get_model_info(self.PROVIDER_NAME, model)

        url = f"{self.API_BASE_URL}/models/{model}:batchEmbedContents"
        payload = self._build_request_payload(
            params.inputs, task_type=task_type, dimensions=params.dimensions, **kwargs
        )
        headers = self._get_headers(api_key)

        start_time = time.time()
        response = self._make_request_with_retry(url, payload, headers)
        latency_ms = self._measure_latency(start_time)

        return self._parse_response(
            response_data=response.json(),
            model=model,
            inputs=params.inputs,
            input_type=params.input_type or "document",
            latency_ms=latency_ms,
            cost_per_million=model_info.cost_per_million_tokens,
        )

    async def aembed(
        self,
        model: str,
        inputs: List[str],
        input_type: Optional[Literal["query", "document"]] = None,
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

        """
        params = self._validate_inputs(
            inputs, input_type=input_type, dimensions=dimensions
        )
        model_info = self.catalog.get_model_info(self.PROVIDER_NAME, model)

        url = f"{self.API_BASE_URL}/models/{model}:batchEmbedContents"
        payload = self._build_request_payload(
            params.inputs, task_type=task_type, dimensions=params.dimensions, **kwargs
        )
        headers = self._get_headers(api_key)

        start_time = time.time()
        response = await self._make_request_with_retry_async(url, payload, headers)
        latency_ms = self._measure_latency(start_time)

        return self._parse_response(
            response_data=response.json(),
            model=model,
            inputs=params.inputs,
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
