"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from mimie.models import EmbedResponse, ModelInfo, TokenizeResponse, Usage


class TestUsage:
    """Tests for Usage model."""

    def test_valid_usage(self):
        """Test creating valid Usage object."""
        usage = Usage(tokens=100, cost=0.00001)
        assert usage.tokens == 100
        assert usage.cost == 0.00001

    def test_negative_tokens(self):
        """Test that negative tokens raises validation error."""
        with pytest.raises(ValidationError):
            Usage(tokens=-1, cost=0.0)

    def test_negative_cost(self):
        """Test that negative cost raises validation error."""
        with pytest.raises(ValidationError):
            Usage(tokens=100, cost=-0.01)


class TestEmbedResponse:
    """Tests for EmbedResponse model."""

    def test_valid_embed_response(self):
        """Test creating valid EmbedResponse."""
        response = EmbedResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            model="voyage-3",
            provider="voyageai",
            dimensions=3,
            usage=Usage(tokens=10, cost=0.0000006),
            latency_ms=123.45,
            input_count=1,
            input_type="query"
        )
        assert len(response.embeddings) == 1
        assert len(response.embeddings[0]) == 3
        assert response.model == "voyage-3"
        assert response.provider == "voyageai"
        assert response.dimensions == 3
        assert response.usage.tokens == 10
        assert response.latency_ms == 123.45

    def test_default_input_type(self):
        """Test default input_type is 'document'."""
        response = EmbedResponse(
            embeddings=[[0.1]],
            model="test",
            provider="test",
            dimensions=1,
            usage=Usage(tokens=1, cost=0.0),
            latency_ms=100.0,
            input_count=1,
        )
        assert response.input_type == "document"

    def test_invalid_input_type(self):
        """Test that invalid input_type raises error."""
        with pytest.raises(ValidationError):
            EmbedResponse(
                embeddings=[[0.1]],
                model="test",
                provider="test",
                dimensions=1,
                usage=Usage(tokens=1, cost=0.0),
                latency_ms=100.0,
                input_count=1,
                input_type="invalid"
            )

    def test_empty_embeddings(self):
        """Test that empty embeddings raises error."""
        with pytest.raises(ValidationError):
            EmbedResponse(
                embeddings=[],
                model="test",
                provider="test",
                dimensions=1,
                usage=Usage(tokens=1, cost=0.0),
                latency_ms=100.0,
                input_count=1,
            )

    def test_inconsistent_embedding_dimensions(self):
        """Test that embeddings with different lengths raises error."""
        with pytest.raises(ValidationError):
            EmbedResponse(
                embeddings=[[0.1, 0.2], [0.1, 0.2, 0.3]],
                model="test",
                provider="test",
                dimensions=2,
                usage=Usage(tokens=1, cost=0.0),
                latency_ms=100.0,
                input_count=2,
            )

    def test_to_numpy(self):
        """Test to_numpy() conversion."""
        response = EmbedResponse(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            model="test",
            provider="test",
            dimensions=2,
            usage=Usage(tokens=1, cost=0.0),
            latency_ms=100.0,
            input_count=2,
        )

        try:
            import numpy as np  # noqa: F401
            arr = response.to_numpy()
            assert arr.shape == (2, 2)
            assert arr[0, 0] == 0.1
        except ImportError:
            pytest.skip("numpy not installed")

    def test_repr(self):
        """Test string representation."""
        response = EmbedResponse(
            embeddings=[[0.1]],
            model="voyage-3",
            provider="voyageai",
            dimensions=1,
            usage=Usage(tokens=10, cost=0.0000006),
            latency_ms=123.45,
            input_count=1,
        )
        repr_str = repr(response)
        assert "voyage-3" in repr_str
        assert "voyageai" in repr_str
        assert "10" in repr_str


class TestModelInfo:
    """Tests for ModelInfo model."""

    def test_valid_model_info(self):
        """Test creating valid ModelInfo."""
        info = ModelInfo(
            name="voyage-3",
            provider="voyageai",
            dimensions=1024,
            max_input_tokens=32000,
            cost_per_million_tokens=0.06,
            supports_batching=True,
            supports_input_type=True,
            supports_dimensions=True,
            tokenizer={"repo": "voyageai/voyage-3"}
        )
        assert info.name == "voyage-3"
        assert info.dimensions == 1024
        assert info.tokenizer["repo"] == "voyageai/voyage-3"

    def test_negative_dimensions(self):
        """Test that negative dimensions raises error."""
        with pytest.raises(ValidationError):
            ModelInfo(
                name="test",
                provider="test",
                dimensions=-1,
                max_input_tokens=1000,
                cost_per_million_tokens=0.1,
            )

    def test_negative_cost(self):
        """Test that negative cost raises error."""
        with pytest.raises(ValidationError):
            ModelInfo(
                name="test",
                provider="test",
                dimensions=100,
                max_input_tokens=1000,
                cost_per_million_tokens=-0.1,
            )


class TestTokenizeResponse:
    """Tests for TokenizeResponse model."""

    def test_valid_tokenize_response(self):
        """Test creating valid TokenizeResponse."""
        response = TokenizeResponse(
            token_count=10,
            model="voyage-3",
            provider="voyageai"
        )
        assert response.token_count == 10
        assert response.model == "voyage-3"
        assert response.tokens is None

    def test_with_tokens(self):
        """Test TokenizeResponse with token IDs."""
        response = TokenizeResponse(
            tokens=[1, 2, 3],
            token_count=3,
            model="voyage-3",
            provider="voyageai"
        )
        assert response.tokens == [1, 2, 3]
        assert response.token_count == 3

    def test_negative_token_count(self):
        """Test that negative token_count raises error."""
        with pytest.raises(ValidationError):
            TokenizeResponse(
                token_count=-1,
                model="test",
                provider="test"
            )
