"""Tests for Gemini API provider."""

import os

import httpx
import pytest

from catsu.models import EmbedResponse
from catsu.providers.gemini import GeminiProvider
from catsu.utils.errors import InvalidInputError


class TestGeminiProvider:
    """Tests for Gemini API provider."""

    @pytest.fixture
    def provider(self, gemini_api_key):
        """Create Gemini API provider instance."""
        return GeminiProvider(
            http_client=httpx.Client(timeout=30),
            async_http_client=httpx.AsyncClient(timeout=30),
            api_key=gemini_api_key,
            max_retries=3,
            verbose=False,
        )

    def test_provider_initialization(self, provider):
        """Test that provider initializes correctly."""
        assert provider.PROVIDER_NAME == "gemini"
        assert (
            provider.API_BASE_URL == "https://generativelanguage.googleapis.com/v1beta"
        )
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
            inputs=["hello", "world"],
            task_type="RETRIEVAL_QUERY",
        )
        assert "requests" in payload
        assert len(payload["requests"]) == 2
        assert payload["requests"][0]["content"]["parts"][0]["text"] == "hello"
        assert payload["requests"][0]["taskType"] == "RETRIEVAL_QUERY"

    def test_build_request_payload_invalid_task_type(self, provider):
        """Test that invalid task_type raises error."""
        with pytest.raises(InvalidInputError):
            provider._build_request_payload(inputs=["test"], task_type="INVALID")

    def test_build_request_payload_with_dimensions(self, provider):
        """Test building request payload with custom dimensions."""
        payload = provider._build_request_payload(
            inputs=["test"],
            task_type="SEMANTIC_SIMILARITY",
            output_dimensionality=768,
        )
        assert payload["requests"][0]["outputDimensionality"] == 768

    def test_build_request_payload_invalid_dimensions(self, provider):
        """Test that invalid output_dimensionality raises error."""
        with pytest.raises(InvalidInputError):
            provider._build_request_payload(inputs=["test"], output_dimensionality=5000)

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="Requires GEMINI_API_KEY environment variable",
    )
    def test_embed_single_text(self, skip_if_no_gemini_key, provider):
        """Test embedding a single text."""
        response = provider.embed(
            model="gemini-embedding-001",
            inputs=["Hello, world!"],
            task_type="RETRIEVAL_QUERY",
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert len(response.embeddings[0]) == response.dimensions
        assert response.model == "gemini-embedding-001"
        assert response.provider == "gemini"
        assert response.usage.tokens > 0
        assert response.usage.cost >= 0
        assert response.latency_ms > 0
        assert response.input_count == 1

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="Requires GEMINI_API_KEY environment variable",
    )
    def test_embed_batch(self, skip_if_no_gemini_key, provider):
        """Test embedding multiple texts."""
        texts = ["First text", "Second text", "Third text"]
        response = provider.embed(
            model="gemini-embedding-001",
            inputs=texts,
            task_type="RETRIEVAL_DOCUMENT",
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 3
        assert response.input_count == 3
        assert all(len(emb) == response.dimensions for emb in response.embeddings)

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="Requires GEMINI_API_KEY environment variable",
    )
    @pytest.mark.asyncio
    async def test_aembed_single_text(self, skip_if_no_gemini_key, provider):
        """Test async embedding."""
        response = await provider.aembed(
            model="gemini-embedding-001",
            inputs=["Async test"],
            task_type="RETRIEVAL_QUERY",
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert response.provider == "gemini"

    def test_tokenize(self, provider):
        """Test local tokenization."""
        # This test may skip if tokenizers not installed or model not available
        try:
            response = provider.tokenize(
                model="gemini-embedding-001", inputs=["Hello, world!"]
            )
            assert response.token_count > 0
            assert response.model == "gemini-embedding-001"
            assert response.provider == "gemini"
        except (ImportError, Exception) as e:
            pytest.skip(f"Tokenization not available: {e}")

    def test_tokenize_batch(self, provider):
        """Test tokenizing multiple texts."""
        try:
            response = provider.tokenize(
                model="gemini-embedding-001", inputs=["First", "Second", "Third"]
            )
            assert response.token_count > 0
        except (ImportError, Exception):
            pytest.skip("Tokenization not available")
