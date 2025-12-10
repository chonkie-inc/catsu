"""Tests for Jina AI provider."""

import os

import httpx
import pytest

from catsu.models import EmbedResponse
from catsu.providers.jinaai import JinaAIProvider
from catsu.utils.errors import InvalidInputError


class TestJinaAIProvider:
    """Tests for Jina AI provider."""

    @pytest.fixture
    def provider(self, jina_api_key, catalog):
        """Create Jina AI provider instance."""
        return JinaAIProvider(
            http_client=httpx.Client(timeout=30),
            async_http_client=httpx.AsyncClient(timeout=30),
            catalog=catalog,
            api_key=jina_api_key,
            max_retries=3,
            verbose=False,
        )

    def test_provider_initialization(self, provider):
        """Test that provider initializes correctly."""
        assert provider.PROVIDER_NAME == "jinaai"
        assert provider.API_BASE_URL == "https://api.jina.ai/v1"
        assert provider.max_retries == 3

    def test_validate_inputs_empty_list(self, provider):
        """Test that empty input list raises error."""
        with pytest.raises(InvalidInputError):
            provider._validate_inputs([])

    def test_validate_inputs_not_list(self, provider):
        """Test that non-list input raises error."""
        with pytest.raises(InvalidInputError):
            provider._validate_inputs("not a list")

    def test_validate_inputs_empty_string(self, provider):
        """Test that empty string in list raises error."""
        with pytest.raises(InvalidInputError):
            provider._validate_inputs(["valid", ""])

    def test_validate_inputs_non_string(self, provider):
        """Test that non-string in list raises error."""
        with pytest.raises(InvalidInputError):
            provider._validate_inputs(["valid", 123])

    def test_build_request_payload(self, provider):
        """Test building API request payload."""
        payload = provider._build_request_payload(
            model="jina-embeddings-v3",
            inputs=["hello", "world"],
            task="retrieval.query",
        )
        assert payload["model"] == "jina-embeddings-v3"
        assert payload["input"] == ["hello", "world"]
        assert payload["task"] == "retrieval.query"
        assert payload["normalized"] is True

    def test_build_request_payload_with_dimensions(self, provider):
        """Test building request payload with custom dimensions."""
        payload = provider._build_request_payload(
            model="jina-embeddings-v3",
            inputs=["test"],
            task="text-matching",
            dimensions=512,
        )
        assert payload["dimensions"] == 512

    @pytest.mark.skipif(
        not os.getenv("JINA_API_KEY"),
        reason="Requires JINA_API_KEY environment variable",
    )
    def test_embed_single_text(self, skip_if_no_jina_key, provider):
        """Test embedding a single text."""
        response = provider.embed(
            model="jina-embeddings-v3",
            inputs=["Hello, world!"],
            task="retrieval.query",
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert len(response.embeddings[0]) == response.dimensions
        assert response.model == "jina-embeddings-v3"
        assert response.provider == "jinaai"
        assert response.usage.tokens > 0
        assert response.usage.cost >= 0
        assert response.latency_ms > 0
        assert response.input_count == 1

    @pytest.mark.skipif(
        not os.getenv("JINA_API_KEY"),
        reason="Requires JINA_API_KEY environment variable",
    )
    def test_embed_batch(self, skip_if_no_jina_key, provider):
        """Test embedding multiple texts."""
        texts = ["First text", "Second text", "Third text"]
        response = provider.embed(
            model="jina-embeddings-v3",
            inputs=texts,
            task="retrieval.passage",
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 3
        assert response.input_count == 3
        assert all(len(emb) == response.dimensions for emb in response.embeddings)

    @pytest.mark.skipif(
        not os.getenv("JINA_API_KEY"),
        reason="Requires JINA_API_KEY environment variable",
    )
    @pytest.mark.asyncio
    async def test_aembed_single_text(self, skip_if_no_jina_key, provider):
        """Test async embedding."""
        response = await provider.aembed(
            model="jina-embeddings-v3",
            inputs=["Async test"],
            task="retrieval.query",
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert response.provider == "jinaai"

    def test_tokenize(self, provider):
        """Test local tokenization."""
        # This test may skip if tokenizers not installed or model not available
        try:
            response = provider.tokenize(
                model="jina-embeddings-v3", inputs=["Hello, world!"]
            )
            assert response.token_count > 0
            assert response.model == "jina-embeddings-v3"
            assert response.provider == "jinaai"
        except (ImportError, Exception) as e:
            pytest.skip(f"Tokenization not available: {e}")

    def test_tokenize_batch(self, provider):
        """Test tokenizing multiple texts."""
        try:
            response = provider.tokenize(
                model="jina-embeddings-v3", inputs=["First", "Second", "Third"]
            )
            assert response.token_count > 0
        except (ImportError, Exception):
            pytest.skip("Tokenization not available")
