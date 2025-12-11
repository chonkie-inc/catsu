"""Tests for Together AI provider."""

import os

import httpx
import pytest

from catsu.models import EmbedResponse
from catsu.providers.togetherai import TogetherAIProvider
from catsu.utils.errors import InvalidInputError


class TestTogetherAIProvider:
    """Tests for Together AI provider."""

    @pytest.fixture
    def provider(self, togetherai_api_key, catalog):
        """Create Together AI provider instance."""
        return TogetherAIProvider(
            http_client=httpx.Client(timeout=30),
            async_http_client=httpx.AsyncClient(timeout=30),
            catalog=catalog,
            api_key=togetherai_api_key,
            max_retries=3,
            verbose=False,
        )

    def test_provider_initialization(self, provider):
        """Test that provider initializes correctly."""
        assert provider.PROVIDER_NAME == "togetherai"
        assert provider.API_BASE_URL == "https://api.together.xyz/v1"
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
            input_type="query",
        )
        assert "input_type" not in payload

    def test_build_request_payload_ignores_dimensions(self, provider):
        """Test that dimensions is ignored in payload."""
        payload = provider._build_request_payload(
            model="BAAI/bge-base-en-v1.5",
            inputs=["test"],
            dimensions=512,
        )
        assert "dimensions" not in payload

    @pytest.mark.skipif(
        not os.getenv("TOGETHER_API_KEY"),
        reason="Requires TOGETHER_API_KEY environment variable",
    )
    def test_embed_single_text_bge(self, skip_if_no_togetherai_key, provider):
        """Test embedding a single text with BGE model."""
        response = provider.embed(
            model="BAAI/bge-base-en-v1.5",
            inputs=["Hello, world!"],
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert len(response.embeddings[0]) == response.dimensions
        assert response.model == "BAAI/bge-base-en-v1.5"
        assert response.provider == "togetherai"
        assert response.usage.tokens > 0
        assert response.usage.cost >= 0
        assert response.latency_ms > 0
        assert response.input_count == 1

    @pytest.mark.skipif(
        not os.getenv("TOGETHER_API_KEY"),
        reason="Requires TOGETHER_API_KEY environment variable",
    )
    def test_embed_multiple_texts_bge(self, skip_if_no_togetherai_key, provider):
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
        not os.getenv("TOGETHER_API_KEY"),
        reason="Requires TOGETHER_API_KEY environment variable",
    )
    def test_embed_uae_model(self, skip_if_no_togetherai_key, provider):
        """Test embedding with UAE-Large-V1 model."""
        response = provider.embed(
            model="WhereIsAI/UAE-Large-V1",
            inputs=["Hello, world!"],
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert response.model == "WhereIsAI/UAE-Large-V1"
        assert response.dimensions == 1024

    @pytest.mark.skipif(
        not os.getenv("TOGETHER_API_KEY"),
        reason="Requires TOGETHER_API_KEY environment variable",
    )
    def test_embed_m2bert_model(self, skip_if_no_togetherai_key, provider):
        """Test embedding with M2-BERT model."""
        response = provider.embed(
            model="togethercomputer/m2-bert-80M-8k-retrieval",
            inputs=["Test sentence for embedding"],
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert response.model == "togethercomputer/m2-bert-80M-8k-retrieval"
        assert response.dimensions == 768

    @pytest.mark.skipif(
        not os.getenv("TOGETHER_API_KEY"),
        reason="Requires TOGETHER_API_KEY environment variable",
    )
    @pytest.mark.asyncio
    async def test_aembed_single_text_bge(self, skip_if_no_togetherai_key, provider):
        """Test async embedding a single text with BGE model."""
        response = await provider.aembed(
            model="BAAI/bge-base-en-v1.5",
            inputs=["Hello, async world!"],
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert response.model == "BAAI/bge-base-en-v1.5"
        assert response.provider == "togetherai"

    @pytest.mark.skipif(
        not os.getenv("TOGETHER_API_KEY"),
        reason="Requires TOGETHER_API_KEY environment variable",
    )
    @pytest.mark.asyncio
    async def test_aembed_multiple_texts(self, skip_if_no_togetherai_key, provider):
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
