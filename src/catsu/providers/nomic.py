"""Nomic embedding provider implementation.

Provides integration with Nomic's embedding API, supporting nomic-embed-text-v1
and nomic-embed-text-v1.5 models with retry logic, cost tracking, and local tokenization.
"""

from typing import Any, Dict, List, Literal, Optional, Set

from ..models import EmbedResponse, TokenizeResponse, Usage
from ..utils import Timer
from ..utils.errors import InvalidInputError
from .base import BaseProvider


class NomicProvider(BaseProvider):
    """Nomic embedding provider.

    Implements the Nomic embeddings API with support for nomic-embed-text-v1
    and nomic-embed-text-v1.5 models.

    Features:
    - Sync and async embedding generation
    - Local tokenization (HuggingFace tokenizers)
    - Automatic retry with exponential backoff
    - Cost and latency tracking
    - Task type specification (search_document, search_query, clustering, classification)
    - Dimension configuration for v1.5 (64-768 dimensions)
    - Long context support (8192 tokens)

    API Documentation: https://docs.nomic.ai/reference/api/embed-text-v-1-embedding-text-post

    """

    API_BASE_URL = "https://api-atlas.nomic.ai/v1"
    PROVIDER_NAME = "nomic"

    VALID_TASK_TYPES: Set[str] = {
        "search_document",
        "search_query",
        "clustering",
        "classification",
    }

    def _map_input_type_to_task_type(self, input_type: Optional[str]) -> str:
        """Map generic input_type to Nomic task_type."""
        if input_type == "query":
            return "search_query"
        return "search_document"

    def _build_request_payload(
        self,
        model: str,
        inputs: List[str],
        task_type: Optional[str] = None,
        dimensions: Optional[int] = None,
        long_text_mode: str = "mean",
        max_tokens_per_text: int = 8192,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build API request payload."""
        payload: Dict[str, Any] = {
            "texts": inputs,  # Nomic uses "texts" instead of "input"
            "model": model,
            "long_text_mode": long_text_mode,
            "max_tokens_per_text": max_tokens_per_text,
        }

        if task_type:
            if task_type not in self.VALID_TASK_TYPES:
                raise InvalidInputError(
                    f"Invalid task_type '{task_type}'. Must be one of: {', '.join(self.VALID_TASK_TYPES)}",
                    parameter="task_type",
                )
            payload["task_type"] = task_type

        # Map to Nomic's "dimensionality" parameter (v1.5 only)
        if dimensions is not None:
            if "v1.5" not in model:
                raise InvalidInputError(
                    "dimensions parameter is only supported for nomic-embed-text-v1.5",
                    parameter="dimensions",
                )
            if not (64 <= dimensions <= 768):
                raise InvalidInputError(
                    "dimensions must be between 64 and 768",
                    parameter="dimensions",
                )
            payload["dimensionality"] = dimensions

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
        embeddings = response_data.get("embeddings", [])
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
        input_type: Optional[Literal["query", "document"]] = None,
        task_type: Optional[str] = None,
        dimensions: Optional[int] = None,
        long_text_mode: str = "mean",
        max_tokens_per_text: int = 8192,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings using Nomic API (synchronous).

        Args:
            model: Model name (e.g., "nomic-embed-text-v1", "nomic-embed-text-v1.5")
            inputs: List of input texts
            input_type: Input type ("query" or "document")
            task_type: Nomic task type (search_document, search_query, clustering, classification)
            dimensions: Output dimensions (v1.5 only, 64-768)
            long_text_mode: How to handle long text ("truncate" or "mean", default: "mean")
            max_tokens_per_text: Maximum tokens per text (default: 8192)
            api_key: Optional API key override for this request
            **kwargs: Additional API parameters

        Returns:
            EmbedResponse with embeddings, usage, and metadata

        """
        params = self._validate_inputs(
            inputs, input_type=input_type, dimensions=dimensions
        )
        model_info = self.catalog.get_model_info(self.PROVIDER_NAME, model)

        # Map input_type to task_type if not provided
        if task_type is None and params.input_type is not None:
            task_type = self._map_input_type_to_task_type(params.input_type)

        url = f"{self.API_BASE_URL}/embedding/text"
        payload = self._build_request_payload(
            model,
            params.inputs,
            task_type=task_type,
            dimensions=params.dimensions,
            long_text_mode=long_text_mode,
            max_tokens_per_text=max_tokens_per_text,
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
        task_type: Optional[str] = None,
        dimensions: Optional[int] = None,
        long_text_mode: str = "mean",
        max_tokens_per_text: int = 8192,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbedResponse:
        """Generate embeddings using Nomic API (asynchronous).

        Args:
            model: Model name (e.g., "nomic-embed-text-v1", "nomic-embed-text-v1.5")
            inputs: List of input texts
            input_type: Input type ("query" or "document")
            task_type: Nomic task type (search_document, search_query, clustering, classification)
            dimensions: Output dimensions (v1.5 only, 64-768)
            long_text_mode: How to handle long text ("truncate" or "mean", default: "mean")
            max_tokens_per_text: Maximum tokens per text (default: 8192)
            api_key: Optional API key override for this request
            **kwargs: Additional API parameters

        Returns:
            EmbedResponse with embeddings, usage, and metadata

        """
        params = self._validate_inputs(
            inputs, input_type=input_type, dimensions=dimensions
        )
        model_info = self.catalog.get_model_info(self.PROVIDER_NAME, model)

        # Map input_type to task_type if not provided
        if task_type is None and params.input_type is not None:
            task_type = self._map_input_type_to_task_type(params.input_type)

        url = f"{self.API_BASE_URL}/embedding/text"
        payload = self._build_request_payload(
            model,
            params.inputs,
            task_type=task_type,
            dimensions=params.dimensions,
            long_text_mode=long_text_mode,
            max_tokens_per_text=max_tokens_per_text,
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
