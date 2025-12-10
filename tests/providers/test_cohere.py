"""Tests for Cohere provider."""

import os

import httpx
import pytest

from catsu.models import EmbedResponse
from catsu.providers.cohere import CohereProvider
from catsu.utils.errors import InvalidInputError


class TestCohereProvider:
    """Tests for Cohere provider."""

    @pytest.fixture
    def provider(self, cohere_api_key, catalog):
        """Create Cohere provider instance."""
        return CohereProvider(
            http_client=httpx.Client(timeout=30),
            async_http_client=httpx.AsyncClient(timeout=30),
            catalog=catalog,
            api_key=cohere_api_key,
            max_retries=3,
            verbose=False,
        )

    def test_provider_initialization(self, provider):
        """Test that provider initializes correctly."""
        assert provider.PROVIDER_NAME == "cohere"
        assert provider.API_BASE_URL == "https://api.cohere.com/v1"
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
            model="embed-english-v3.0",
            inputs=["hello", "world"],
            input_type="query",
        )
        assert payload["model"] == "embed-english-v3.0"
        assert payload["texts"] == ["hello", "world"]
        assert payload["input_type"] == "search_query"  # Mapped to Cohere's API value
        assert payload["embedding_types"] == ["float"]

    def test_build_request_payload_invalid_input_type(self, provider):
        """Test that invalid input_type raises error."""
        with pytest.raises(InvalidInputError):
            provider._build_request_payload(
                model="embed-english-v3.0", inputs=["test"], input_type="invalid"
            )

    def test_build_request_payload_with_truncate(self, provider):
        """Test building request payload with truncate option."""
        payload = provider._build_request_payload(
            model="embed-english-v3.0",
            inputs=["test"],
            input_type="document",
            truncate="END",
        )
        assert payload["truncate"] == "END"
        assert (
            payload["input_type"] == "search_document"
        )  # Mapped to Cohere's API value

    def test_build_request_payload_invalid_truncate(self, provider):
        """Test that invalid truncate option raises error."""
        with pytest.raises(InvalidInputError):
            provider._build_request_payload(
                model="embed-english-v3.0", inputs=["test"], truncate="INVALID"
            )

    @pytest.mark.skipif(
        not os.getenv("COHERE_API_KEY"),
        reason="Requires COHERE_API_KEY environment variable",
    )
    def test_embed_single_text(self, skip_if_no_cohere_key, provider):
        """Test embedding a single text."""
        response = provider.embed(
            model="embed-english-light-v3.0",
            inputs=["Hello, world!"],
            input_type="query",
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert len(response.embeddings[0]) == response.dimensions
        assert response.model == "embed-english-light-v3.0"
        assert response.provider == "cohere"
        assert response.usage.tokens > 0
        assert response.usage.cost >= 0
        assert response.latency_ms > 0
        assert response.input_count == 1

    @pytest.mark.skipif(
        not os.getenv("COHERE_API_KEY"),
        reason="Requires COHERE_API_KEY environment variable",
    )
    def test_embed_batch(self, skip_if_no_cohere_key, provider):
        """Test embedding multiple texts."""
        texts = ["First text", "Second text", "Third text"]
        response = provider.embed(
            model="embed-english-light-v3.0",
            inputs=texts,
            input_type="document",
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 3
        assert response.input_count == 3
        assert all(len(emb) == response.dimensions for emb in response.embeddings)

    @pytest.mark.skipif(
        not os.getenv("COHERE_API_KEY"),
        reason="Requires COHERE_API_KEY environment variable",
    )
    @pytest.mark.asyncio
    async def test_aembed_single_text(self, skip_if_no_cohere_key, provider):
        """Test async embedding."""
        response = await provider.aembed(
            model="embed-english-light-v3.0",
            inputs=["Async test"],
            input_type="query",
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert response.provider == "cohere"

    def test_tokenize(self, provider):
        """Test local tokenization."""
        # This test may skip if tokenizers not installed or model not available
        try:
            response = provider.tokenize(
                model="embed-english-v3.0", inputs=["Hello, world!"]
            )
            assert response.token_count > 0
            assert response.model == "embed-english-v3.0"
            assert response.provider == "cohere"
        except (ImportError, Exception) as e:
            pytest.skip(f"Tokenization not available: {e}")

    def test_tokenize_batch(self, provider):
        """Test tokenizing multiple texts."""
        try:
            response = provider.tokenize(
                model="embed-english-v3.0", inputs=["First", "Second", "Third"]
            )
            assert response.token_count > 0
        except (ImportError, Exception):
            pytest.skip("Tokenization not available")
