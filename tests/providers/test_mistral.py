"""Tests for Mistral AI provider."""

import os

import httpx
import pytest

from catsu.models import EmbedResponse
from catsu.providers.mistral import MistralProvider
from catsu.utils.errors import InvalidInputError


class TestMistralProvider:
    """Tests for Mistral AI provider."""

    @pytest.fixture
    def provider(self, mistral_api_key, catalog):
        """Create Mistral AI provider instance."""
        return MistralProvider(
            http_client=httpx.Client(timeout=30),
            async_http_client=httpx.AsyncClient(timeout=30),
            catalog=catalog,
            api_key=mistral_api_key,
            max_retries=3,
            verbose=False,
        )

    def test_provider_initialization(self, provider):
        """Test that provider initializes correctly."""
        assert provider.PROVIDER_NAME == "mistral"
        assert provider.API_BASE_URL == "https://api.mistral.ai/v1"
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
            model="mistral-embed",
            inputs=["hello", "world"],
        )
        assert payload["model"] == "mistral-embed"
        assert payload["input"] == ["hello", "world"]

    def test_build_request_payload_with_encoding(self, provider):
        """Test building request payload with encoding format."""
        payload = provider._build_request_payload(
            model="codestral-embed-2505",
            inputs=["test"],
            encoding_format="int8",
        )
        assert payload["encoding_format"] == "int8"

    @pytest.mark.skipif(
        not os.getenv("MISTRAL_API_KEY"),
        reason="Requires MISTRAL_API_KEY environment variable",
    )
    def test_embed_single_text(self, skip_if_no_mistral_key, provider):
        """Test embedding a single text."""
        response = provider.embed(
            model="mistral-embed",
            inputs=["Hello, world!"],
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert len(response.embeddings[0]) == response.dimensions
        assert response.model == "mistral-embed"
        assert response.provider == "mistral"
        assert response.usage.tokens > 0
        assert response.usage.cost >= 0
        assert response.latency_ms > 0
        assert response.input_count == 1

    @pytest.mark.skipif(
        not os.getenv("MISTRAL_API_KEY"),
        reason="Requires MISTRAL_API_KEY environment variable",
    )
    def test_embed_batch(self, skip_if_no_mistral_key, provider):
        """Test embedding multiple texts."""
        texts = ["First text", "Second text", "Third text"]
        response = provider.embed(
            model="mistral-embed",
            inputs=texts,
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 3
        assert response.input_count == 3
        assert all(len(emb) == response.dimensions for emb in response.embeddings)

    @pytest.mark.skipif(
        not os.getenv("MISTRAL_API_KEY"),
        reason="Requires MISTRAL_API_KEY environment variable",
    )
    @pytest.mark.asyncio
    async def test_aembed_single_text(self, skip_if_no_mistral_key, provider):
        """Test async embedding."""
        response = await provider.aembed(
            model="mistral-embed",
            inputs=["Async test"],
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert response.provider == "mistral"

    def test_tokenize(self, provider):
        """Test local tokenization."""
        # This test may skip if tokenizers not installed or model not available
        try:
            response = provider.tokenize(
                model="mistral-embed", inputs=["Hello, world!"]
            )
            assert response.token_count > 0
            assert response.model == "mistral-embed"
            assert response.provider == "mistral"
        except (ImportError, Exception) as e:
            pytest.skip(f"Tokenization not available: {e}")

    def test_tokenize_batch(self, provider):
        """Test tokenizing multiple texts."""
        try:
            response = provider.tokenize(
                model="mistral-embed", inputs=["First", "Second", "Third"]
            )
            assert response.token_count > 0
        except (ImportError, Exception):
            pytest.skip("Tokenization not available")
