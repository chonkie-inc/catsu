"""Pydantic models for Catsu API responses and data structures.

This module defines the response models and data structures used throughout
the Catsu library, providing type safety and validation.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    try:
        import numpy as np  # type: ignore
    except ImportError:
        np = None  # type: ignore


class Usage(BaseModel):
    """Token usage and cost information for an embedding request.

    Attributes:
        tokens: Total number of tokens processed
        cost: Total cost in USD

    """

    tokens: int = Field(..., description="Total tokens processed")
    cost: float = Field(..., description="Total cost in USD")

    @field_validator("tokens")
    @classmethod
    def validate_tokens(cls, v: int) -> int:
        """Validate that tokens is non-negative."""
        if v < 0:
            raise ValueError("tokens must be non-negative")
        return v

    @field_validator("cost")
    @classmethod
    def validate_cost(cls, v: float) -> float:
        """Validate that cost is non-negative."""
        if v < 0:
            raise ValueError("cost must be non-negative")
        return v


class EmbedResponse(BaseModel):
    """Response from an embedding request.

    Contains embeddings, metadata, usage information, and latency tracking.

    Attributes:
        embeddings: List of embedding vectors (always list of lists)
        model: Model name used (e.g., "voyage-3")
        provider: Provider name (e.g., "voyageai")
        dimensions: Dimensionality of embeddings (e.g., 1024)
        usage: Token usage and cost information
        latency_ms: Request latency in milliseconds
        input_count: Number of input texts processed
        input_type: Type of input ("query" or "document")

    Example:
        >>> response = EmbedResponse(
        ...     embeddings=[[0.1, 0.2, 0.3]],
        ...     model="voyage-3",
        ...     provider="voyageai",
        ...     dimensions=3,
        ...     usage=Usage(tokens=10, cost=0.000001),
        ...     latency_ms=123.45,
        ...     input_count=1,
        ...     input_type="query"
        ... )
        >>> print(response.embeddings[0][:3])
        [0.1, 0.2, 0.3]

    """

    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    model: str = Field(..., description="Model name")
    provider: str = Field(..., description="Provider name")
    dimensions: int = Field(..., description="Embedding dimensionality")
    usage: Usage = Field(..., description="Usage and cost information")
    latency_ms: float = Field(..., description="Request latency in milliseconds")
    input_count: int = Field(..., description="Number of inputs processed")
    input_type: str = Field(
        default="document",
        description='Input type: "query" or "document"',
    )

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        """Validate that dimensions is positive."""
        if v <= 0:
            raise ValueError("dimensions must be positive")
        return v

    @field_validator("latency_ms")
    @classmethod
    def validate_latency(cls, v: float) -> float:
        """Validate that latency is non-negative."""
        if v < 0:
            raise ValueError("latency_ms must be non-negative")
        return v

    @field_validator("input_count")
    @classmethod
    def validate_input_count(cls, v: int) -> int:
        """Validate that input_count is positive."""
        if v <= 0:
            raise ValueError("input_count must be positive")
        return v

    @field_validator("input_type")
    @classmethod
    def validate_input_type(cls, v: str) -> str:
        """Validate that input_type is either 'query' or 'document'."""
        allowed = {"query", "document"}
        if v not in allowed:
            raise ValueError(f"input_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("embeddings")
    @classmethod
    def validate_embeddings(cls, v: List[List[float]]) -> List[List[float]]:
        """Validate embeddings structure."""
        if not v:
            raise ValueError("embeddings cannot be empty")
        if not all(isinstance(emb, list) for emb in v):
            raise ValueError("embeddings must be a list of lists")
        # Check all embeddings have same length
        if len(v) > 1:
            first_len = len(v[0])
            if not all(len(emb) == first_len for emb in v):
                raise ValueError("all embeddings must have the same length")
        return v

    def to_numpy(self) -> "np.ndarray":
        """Convert embeddings to numpy array.

        Requires numpy to be installed (catsu[numpy]).

        Returns:
            numpy.ndarray of shape (input_count, dimensions)

        Raises:
            ImportError: If numpy is not installed

        Example:
            >>> response = client.embed(model="voyage-3", input="hello")
            >>> embeddings_array = response.to_numpy()
            >>> print(embeddings_array.shape)
            (1, 1024)

        """
        try:
            import numpy as np  # type: ignore

            return np.array(self.embeddings)
        except ImportError as e:
            raise ImportError(
                "numpy is required for to_numpy(). "
                "Install it with: pip install catsu[numpy]"
            ) from e

    def __repr__(self) -> str:
        """Return string representation of EmbedResponse."""
        return (
            f"EmbedResponse(model='{self.model}', provider='{self.provider}', "
            f"input_count={self.input_count}, dimensions={self.dimensions}, "
            f"tokens={self.usage.tokens}, cost=${self.usage.cost:.6f}, "
            f"latency={self.latency_ms:.2f}ms)"
        )


class ModelInfo(BaseModel):
    """Information about an embedding model.

    Used by the model catalog to store and retrieve model metadata.

    Attributes:
        name: Model name (e.g., "voyage-3")
        provider: Provider name (e.g., "voyageai")
        dimensions: Default embedding dimensions
        max_input_tokens: Maximum tokens per request
        cost_per_million_tokens: Cost per million tokens in USD
        supports_batching: Whether batch requests are supported
        supports_input_type: Whether input_type parameter is supported
        supports_dimensions: Whether custom dimensions are supported
        tokenizer: Tokenizer configuration (e.g., {"repo": "voyageai/voyage-3"})
        description: Human-readable model description

    """

    name: str = Field(..., description="Model name")
    provider: str = Field(..., description="Provider name")
    dimensions: int = Field(..., description="Default embedding dimensions")
    max_input_tokens: int = Field(..., description="Max tokens per request")
    cost_per_million_tokens: float = Field(
        ..., description="Cost per million tokens in USD"
    )
    supports_batching: bool = Field(default=True, description="Supports batch requests")
    supports_input_type: bool = Field(
        default=False, description="Supports input_type parameter"
    )
    supports_dimensions: bool = Field(
        default=False, description="Supports custom dimensions"
    )
    tokenizer: Optional[Dict[str, Any]] = Field(
        default=None, description="Tokenizer configuration"
    )
    description: Optional[str] = Field(default=None, description="Model description")

    @field_validator("dimensions", "max_input_tokens")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Validate that integer fields are positive."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("cost_per_million_tokens")
    @classmethod
    def validate_cost(cls, v: float) -> float:
        """Validate that cost is non-negative."""
        if v < 0:
            raise ValueError("cost_per_million_tokens must be non-negative")
        return v

    def __repr__(self) -> str:
        """Return string representation of ModelInfo."""
        return (
            f"ModelInfo(name='{self.name}', provider='{self.provider}', "
            f"dimensions={self.dimensions}, cost=${self.cost_per_million_tokens}/M)"
        )


class TokenizeResponse(BaseModel):
    """Response from a tokenization request.

    Used for counting tokens without performing embedding.

    Attributes:
        tokens: List of token IDs (if available)
        token_count: Total number of tokens
        model: Model name used for tokenization
        provider: Provider name

    Example:
        >>> response = client.tokenize(model="voyage-3", input="hello world")
        >>> print(response.token_count)
        2

    """

    tokens: Optional[List[int]] = Field(
        default=None, description="Token IDs (if available)"
    )
    token_count: int = Field(..., description="Total number of tokens")
    model: str = Field(..., description="Model name")
    provider: str = Field(..., description="Provider name")

    @field_validator("token_count")
    @classmethod
    def validate_token_count(cls, v: int) -> int:
        """Validate that token_count is non-negative."""
        if v < 0:
            raise ValueError("token_count must be non-negative")
        return v

    def __repr__(self) -> str:
        """Return string representation of TokenizeResponse."""
        return (
            f"TokenizeResponse(model='{self.model}', provider='{self.provider}', "
            f"token_count={self.token_count})"
        )
