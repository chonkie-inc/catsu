"""Tests for DeepInfra provider."""

import os

import httpx
import pytest

from catsu.models import EmbedResponse
from catsu.providers.deepinfra import DeepInfraProvider
from catsu.utils.errors import InvalidInputError


class TestDeepInfraProvider:
    """Tests for DeepInfra provider."""

    @pytest.fixture
    def provider(self, deepinfra_api_key, catalog):
        """Create DeepInfra provider instance."""
        return DeepInfraProvider(
            http_client=httpx.Client(timeout=30),
            async_http_client=httpx.AsyncClient(timeout=30),
            catalog=catalog,
            api_key=deepinfra_api_key,
            max_retries=3,
            verbose=False,
        )

    def test_provider_initialization(self, provider):
        """Test that provider initializes correctly."""
        assert provider.PROVIDER_NAME == "deepinfra"
        assert provider.API_BASE_URL == "https://api.deepinfra.com/v1/openai"
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

    def test_build_request_payload_basic(self, provider):
        """Test building basic API request payload."""
        payload = provider._build_request_payload(
            model="BAAI/bge-base-en-v1.5",
            inputs=["hello", "world"],
        )
        assert payload["model"] == "BAAI/bge-base-en-v1.5"
        assert payload["input"] == ["hello", "world"]

    def test_build_request_payload_ignores_input_type(self, provider):
        """Test that input_type is ignored in payload."""
        payload = provider._build_request_payload(
            model="BAAI/bge-base-en-v1.5",
            inputs=["test"],
            input_type="query",  # Should be ignored
        )
        assert "input_type" not in payload

    def test_build_request_payload_with_dimensions(self, provider):
        """Test that dimensions is included in payload for supported models."""
        payload = provider._build_request_payload(
            model="Qwen/Qwen3-Embedding-8B",
            inputs=["test"],
            dimensions=512,
        )
        assert payload["dimensions"] == 512

    def test_build_request_payload_without_dimensions(self, provider):
        """Test that dimensions is not included when not provided."""
        payload = provider._build_request_payload(
            model="BAAI/bge-base-en-v1.5",
            inputs=["test"],
        )
        assert "dimensions" not in payload

    @pytest.mark.skipif(
        not os.getenv("DEEPINFRA_API_KEY"),
        reason="Requires DEEPINFRA_API_KEY environment variable",
    )
    def test_embed_single_text_bge(self, skip_if_no_deepinfra_key, provider):
        """Test embedding a single text with BGE model."""
        response = provider.embed(
            model="BAAI/bge-base-en-v1.5",
            inputs=["Hello, world!"],
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert len(response.embeddings[0]) == response.dimensions
        assert response.model == "BAAI/bge-base-en-v1.5"
        assert response.provider == "deepinfra"
        assert response.usage.tokens > 0
        assert response.usage.cost >= 0
        assert response.latency_ms > 0
        assert response.input_count == 1

    @pytest.mark.skipif(
        not os.getenv("DEEPINFRA_API_KEY"),
        reason="Requires DEEPINFRA_API_KEY environment variable",
    )
    def test_embed_multiple_texts_bge(self, skip_if_no_deepinfra_key, provider):
        """Test embedding multiple texts with BGE model."""
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is transforming technology",
            "Python is a versatile programming language",
        ]
        response = provider.embed(
            model="BAAI/bge-base-en-v1.5",
            inputs=texts,
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 3
        assert response.input_count == 3
        assert all(len(emb) == response.dimensions for emb in response.embeddings)

    @pytest.mark.skipif(
        not os.getenv("DEEPINFRA_API_KEY"),
        reason="Requires DEEPINFRA_API_KEY environment variable",
    )
    def test_embed_e5_model(self, skip_if_no_deepinfra_key, provider):
        """Test embedding with E5 model."""
        response = provider.embed(
            model="intfloat/e5-base-v2",
            inputs=["Hello, world!"],
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert response.model == "intfloat/e5-base-v2"
        assert response.dimensions == 768  # E5-base-v2 has 768 dimensions

    @pytest.mark.skipif(
        not os.getenv("DEEPINFRA_API_KEY"),
        reason="Requires DEEPINFRA_API_KEY environment variable",
    )
    def test_embed_minilm_model(self, skip_if_no_deepinfra_key, provider):
        """Test embedding with MiniLM model."""
        response = provider.embed(
            model="sentence-transformers/all-MiniLM-L6-v2",
            inputs=["Test sentence for embedding"],
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert response.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert response.dimensions == 384  # MiniLM has 384 dimensions

    @pytest.mark.skipif(
        not os.getenv("DEEPINFRA_API_KEY"),
        reason="Requires DEEPINFRA_API_KEY environment variable",
    )
    @pytest.mark.asyncio
    async def test_aembed_single_text_bge(self, skip_if_no_deepinfra_key, provider):
        """Test async embedding a single text with BGE model."""
        response = await provider.aembed(
            model="BAAI/bge-base-en-v1.5",
            inputs=["Hello, async world!"],
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert response.model == "BAAI/bge-base-en-v1.5"
        assert response.provider == "deepinfra"

    @pytest.mark.skipif(
        not os.getenv("DEEPINFRA_API_KEY"),
        reason="Requires DEEPINFRA_API_KEY environment variable",
    )
    @pytest.mark.asyncio
    async def test_aembed_multiple_texts(self, skip_if_no_deepinfra_key, provider):
        """Test async embedding multiple texts."""
        texts = [
            "First async text",
            "Second async text",
        ]
        response = await provider.aembed(
            model="BAAI/bge-base-en-v1.5",
            inputs=texts,
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 2
        assert response.input_count == 2
