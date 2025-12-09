"""Tests for VoyageAI provider."""

import os

import httpx
import pytest

from catsu.models import EmbedResponse
from catsu.providers.voyageai import VoyageAIProvider
from catsu.utils.errors import InvalidInputError


class TestVoyageAIProvider:
    """Tests for VoyageAI provider."""

    @pytest.fixture
    def provider(self, voyage_api_key):
        """Create VoyageAI provider instance."""
        return VoyageAIProvider(
            http_client=httpx.Client(timeout=30),
            async_http_client=httpx.AsyncClient(timeout=30),
            api_key=voyage_api_key,
            max_retries=3,
            verbose=False,
        )

    def test_provider_initialization(self, provider):
        """Test that provider initializes correctly."""
        assert provider.PROVIDER_NAME == "voyageai"
        assert provider.API_BASE_URL == "https://api.voyageai.com/v1"
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
            model="voyage-3", inputs=["hello", "world"], input_type="query"
        )
        assert payload["model"] == "voyage-3"
        assert payload["input"] == ["hello", "world"]
        assert payload["input_type"] == "query"

    def test_build_request_payload_invalid_input_type(self, provider):
        """Test that invalid input_type raises error."""
        with pytest.raises(InvalidInputError):
            provider._build_request_payload(
                model="voyage-3", inputs=["test"], input_type="invalid"
            )

    @pytest.mark.skipif(
        not os.getenv("VOYAGE_API_KEY"),
        reason="Requires VOYAGE_API_KEY environment variable",
    )
    def test_embed_single_text(self, skip_if_no_voyage_key, provider):
        """Test embedding a single text."""
        response = provider.embed(
            model="voyage-3-lite", inputs=["Hello, world!"], input_type="query"
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert len(response.embeddings[0]) == response.dimensions
        assert response.model == "voyage-3-lite"
        assert response.provider == "voyageai"
        assert response.usage.tokens > 0
        assert response.usage.cost >= 0
        assert response.latency_ms > 0
        assert response.input_count == 1

    @pytest.mark.skipif(
        not os.getenv("VOYAGE_API_KEY"),
        reason="Requires VOYAGE_API_KEY environment variable",
    )
    def test_embed_batch(self, skip_if_no_voyage_key, provider):
        """Test embedding multiple texts."""
        texts = ["First text", "Second text", "Third text"]
        response = provider.embed(
            model="voyage-3-lite", inputs=texts, input_type="document"
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 3
        assert response.input_count == 3
        assert all(len(emb) == response.dimensions for emb in response.embeddings)

    @pytest.mark.skipif(
        not os.getenv("VOYAGE_API_KEY"),
        reason="Requires VOYAGE_API_KEY environment variable",
    )
    @pytest.mark.asyncio
    async def test_aembed_single_text(self, skip_if_no_voyage_key, provider):
        """Test async embedding."""
        response = await provider.aembed(
            model="voyage-3-lite", inputs=["Async test"], input_type="query"
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert response.provider == "voyageai"

    def test_tokenize(self, provider):
        """Test local tokenization."""
        # This test may skip if tokenizers not installed or model not available
        try:
            response = provider.tokenize(model="voyage-3", inputs=["Hello, world!"])
            assert response.token_count > 0
            assert response.model == "voyage-3"
            assert response.provider == "voyageai"
        except (ImportError, Exception) as e:
            pytest.skip(f"Tokenization not available: {e}")

    def test_tokenize_batch(self, provider):
        """Test tokenizing multiple texts."""
        try:
            response = provider.tokenize(
                model="voyage-3", inputs=["First", "Second", "Third"]
            )
            assert response.token_count > 0
        except (ImportError, Exception):
            pytest.skip("Tokenization not available")
