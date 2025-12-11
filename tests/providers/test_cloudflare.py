"""Tests for Cloudflare Workers AI provider."""

import os

import httpx
import pytest

from catsu.models import EmbedResponse
from catsu.providers.cloudflare import CloudflareProvider
from catsu.utils.errors import AuthenticationError, InvalidInputError


class TestCloudflareProvider:
    """Tests for Cloudflare Workers AI provider."""

    @pytest.fixture
    def provider(self, cloudflare_api_key, cloudflare_account_id, catalog):
        """Create Cloudflare provider instance."""
        return CloudflareProvider(
            http_client=httpx.Client(timeout=30),
            async_http_client=httpx.AsyncClient(timeout=30),
            catalog=catalog,
            api_key=cloudflare_api_key,
            account_id=cloudflare_account_id,
            max_retries=3,
            verbose=False,
        )

    def test_provider_initialization(self, provider):
        """Test that provider initializes correctly."""
        assert provider.PROVIDER_NAME == "cloudflare"
        assert provider.API_BASE_URL == "https://api.cloudflare.com/client/v4/accounts"
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
            model="@cf/baai/bge-base-en-v1.5",
            inputs=["hello", "world"],
        )
        assert payload["text"] == ["hello", "world"]

    def test_build_request_payload_with_pooling(self, provider):
        """Test building request payload with pooling."""
        payload = provider._build_request_payload(
            model="@cf/baai/bge-base-en-v1.5",
            inputs=["test"],
            pooling="cls",
        )
        assert payload["pooling"] == "cls"

    def test_build_request_payload_without_pooling(self, provider):
        """Test building request payload without pooling."""
        payload = provider._build_request_payload(
            model="@cf/baai/bge-base-en-v1.5",
            inputs=["test"],
        )
        assert "pooling" not in payload

    def test_get_account_id_missing(self, catalog):
        """Test that missing account ID raises error."""
        provider = CloudflareProvider(
            http_client=httpx.Client(timeout=30),
            async_http_client=httpx.AsyncClient(timeout=30),
            catalog=catalog,
            api_key="test-key",
            account_id=None,
            max_retries=3,
            verbose=False,
        )
        with pytest.raises(AuthenticationError):
            provider._get_account_id()

    @pytest.mark.skipif(
        not os.getenv("CLOUDFLARE_API_KEY") or not os.getenv("CLOUDFLARE_ACCOUNT_ID"),
        reason="Requires CLOUDFLARE_API_KEY and CLOUDFLARE_ACCOUNT_ID environment variables",
    )
    def test_embed_single_text_bge(self, skip_if_no_cloudflare_key, provider):
        """Test embedding a single text with BGE model."""
        response = provider.embed(
            model="@cf/baai/bge-base-en-v1.5",
            inputs=["Hello, world!"],
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert len(response.embeddings[0]) == response.dimensions
        assert response.model == "@cf/baai/bge-base-en-v1.5"
        assert response.provider == "cloudflare"
        assert response.usage.tokens > 0
        assert response.usage.cost >= 0
        assert response.latency_ms > 0
        assert response.input_count == 1

    @pytest.mark.skipif(
        not os.getenv("CLOUDFLARE_API_KEY") or not os.getenv("CLOUDFLARE_ACCOUNT_ID"),
        reason="Requires CLOUDFLARE_API_KEY and CLOUDFLARE_ACCOUNT_ID environment variables",
    )
    def test_embed_multiple_texts_bge(self, skip_if_no_cloudflare_key, provider):
        """Test embedding multiple texts with BGE model."""
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is transforming technology",
            "Python is a versatile programming language",
        ]
        response = provider.embed(
            model="@cf/baai/bge-base-en-v1.5",
            inputs=texts,
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 3
        assert response.input_count == 3
        assert all(len(emb) == response.dimensions for emb in response.embeddings)

    @pytest.mark.skipif(
        not os.getenv("CLOUDFLARE_API_KEY") or not os.getenv("CLOUDFLARE_ACCOUNT_ID"),
        reason="Requires CLOUDFLARE_API_KEY and CLOUDFLARE_ACCOUNT_ID environment variables",
    )
    def test_embed_with_pooling(self, skip_if_no_cloudflare_key, provider):
        """Test embedding with cls pooling."""
        response = provider.embed(
            model="@cf/baai/bge-base-en-v1.5",
            inputs=["Test text"],
            pooling="cls",
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1

    @pytest.mark.skipif(
        not os.getenv("CLOUDFLARE_API_KEY") or not os.getenv("CLOUDFLARE_ACCOUNT_ID"),
        reason="Requires CLOUDFLARE_API_KEY and CLOUDFLARE_ACCOUNT_ID environment variables",
    )
    @pytest.mark.asyncio
    async def test_aembed_single_text(self, skip_if_no_cloudflare_key, provider):
        """Test async embedding a single text."""
        response = await provider.aembed(
            model="@cf/baai/bge-base-en-v1.5",
            inputs=["Hello, async world!"],
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert response.model == "@cf/baai/bge-base-en-v1.5"
        assert response.provider == "cloudflare"

    @pytest.mark.skipif(
        not os.getenv("CLOUDFLARE_API_KEY") or not os.getenv("CLOUDFLARE_ACCOUNT_ID"),
        reason="Requires CLOUDFLARE_API_KEY and CLOUDFLARE_ACCOUNT_ID environment variables",
    )
    @pytest.mark.asyncio
    async def test_aembed_multiple_texts(self, skip_if_no_cloudflare_key, provider):
        """Test async embedding multiple texts."""
        texts = [
            "First async text",
            "Second async text",
        ]
        response = await provider.aembed(
            model="@cf/baai/bge-base-en-v1.5",
            inputs=texts,
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 2
        assert response.input_count == 2
