"""Tests for Client class."""

import os

import pytest

from mimie import Client
from mimie.models import EmbedResponse
from mimie.utils.errors import (
    InvalidInputError,
    ModelNotFoundError,
)


class TestClientInitialization:
    """Tests for Client initialization."""

    def test_client_init_defaults(self):
        """Test client initializes with default parameters."""
        client = Client()
        assert client.verbose is False
        assert client.max_retries == 3
        assert client.timeout == 30
        assert "voyageai" in client._providers

    def test_client_init_custom_params(self):
        """Test client initializes with custom parameters."""
        client = Client(
            verbose=True, max_retries=5, timeout=60, api_keys={"voyageai": "test-key"}
        )
        assert client.verbose is True
        assert client.max_retries == 5
        assert client.timeout == 60

    def test_client_providers_loaded(self):
        """Test that providers are loaded on init."""
        client = Client()
        assert len(client._providers) > 0
        assert "voyageai" in client._providers

    def test_client_catalog_loaded(self):
        """Test that catalog is loaded on init."""
        client = Client()
        assert client._catalog is not None


class TestClientAPIKeyManagement:
    """Tests for API key management."""

    def test_get_api_key_from_params(self):
        """Test getting API key from constructor params."""
        client = Client(api_keys={"voyageai": "param-key"})
        key = client._get_api_key("voyageai")
        assert key == "param-key"

    def test_get_api_key_from_env(self, monkeypatch):
        """Test getting API key from environment."""
        monkeypatch.setenv("VOYAGE_API_KEY", "env-key")
        client = Client()
        key = client._get_api_key("voyageai")
        assert key == "env-key"

    def test_get_api_key_params_override_env(self, monkeypatch):
        """Test that params override environment."""
        monkeypatch.setenv("VOYAGE_API_KEY", "env-key")
        client = Client(api_keys={"voyageai": "param-key"})
        key = client._get_api_key("voyageai")
        assert key == "param-key"

    def test_get_api_key_unknown_provider(self):
        """Test getting API key for unknown provider."""
        client = Client()
        key = client._get_api_key("unknown-provider")
        assert key is None


class TestClientProviderParsing:
    """Tests for provider parsing logic."""

    def test_parse_model_with_prefix(self):
        """Test parsing model with provider prefix."""
        client = Client()
        provider, model = client._parse_model_string("voyageai:voyage-3")
        assert provider == "voyageai"
        assert model == "voyage-3"

    def test_parse_model_with_explicit_provider(self):
        """Test parsing model with explicit provider param."""
        client = Client()
        provider, model = client._parse_model_string("voyage-3", provider="voyageai")
        assert provider == "voyageai"
        assert model == "voyage-3"

    def test_parse_model_auto_detect(self):
        """Test auto-detecting provider."""
        client = Client()
        provider, model = client._parse_model_string("voyage-3")
        assert provider == "voyageai"
        assert model == "voyage-3"

    def test_parse_model_not_found(self):
        """Test parsing non-existent model."""
        client = Client()
        with pytest.raises(ModelNotFoundError):
            client._parse_model_string("nonexistent-model-xyz")

    def test_parse_model_provider_mismatch(self):
        """Test provider mismatch raises error."""
        client = Client()
        with pytest.raises(InvalidInputError) as exc_info:
            client._parse_model_string("voyageai:voyage-3", provider="openai")
        assert exc_info.value.parameter == "provider"


class TestClientEmbedding:
    """Tests for embedding functionality."""

    @pytest.mark.skipif(
        not os.getenv("VOYAGE_API_KEY"),
        reason="Requires VOYAGE_API_KEY environment variable",
    )
    def test_embed_with_explicit_provider(self, skip_if_no_voyage_key):
        """Test embedding with explicit provider."""
        client = Client()
        response = client.embed(
            provider="voyageai", model="voyage-3-lite", input="Test text"
        )
        assert isinstance(response, EmbedResponse)
        assert response.provider == "voyageai"
        assert len(response.embeddings) == 1

    @pytest.mark.skipif(
        not os.getenv("VOYAGE_API_KEY"),
        reason="Requires VOYAGE_API_KEY environment variable",
    )
    def test_embed_with_prefix(self, skip_if_no_voyage_key):
        """Test embedding with provider prefix."""
        client = Client()
        response = client.embed(model="voyageai:voyage-3-lite", input="Test text")
        assert isinstance(response, EmbedResponse)
        assert response.provider == "voyageai"

    @pytest.mark.skipif(
        not os.getenv("VOYAGE_API_KEY"),
        reason="Requires VOYAGE_API_KEY environment variable",
    )
    def test_embed_with_auto_detect(self, skip_if_no_voyage_key):
        """Test embedding with auto-detection."""
        client = Client()
        response = client.embed(model="voyage-3-lite", input="Test text")
        assert isinstance(response, EmbedResponse)
        assert response.provider == "voyageai"

    @pytest.mark.skipif(
        not os.getenv("VOYAGE_API_KEY"),
        reason="Requires VOYAGE_API_KEY environment variable",
    )
    def test_embed_batch(self, skip_if_no_voyage_key):
        """Test batch embedding."""
        client = Client()
        response = client.embed(
            model="voyage-3-lite", input=["First", "Second", "Third"]
        )
        assert len(response.embeddings) == 3
        assert response.input_count == 3

    @pytest.mark.skipif(
        not os.getenv("VOYAGE_API_KEY"),
        reason="Requires VOYAGE_API_KEY environment variable",
    )
    @pytest.mark.asyncio
    async def test_aembed(self, skip_if_no_voyage_key):
        """Test async embedding."""
        client = Client()
        response = await client.aembed(model="voyage-3-lite", input="Async test")
        assert isinstance(response, EmbedResponse)
        await client.aclose()


class TestClientListModels:
    """Tests for list_models functionality."""

    def test_list_all_models(self):
        """Test listing all models."""
        client = Client()
        models = client.list_models()
        assert len(models) > 0
        assert all("name" in m for m in models)

    def test_list_models_by_provider(self):
        """Test listing models for specific provider."""
        client = Client()
        models = client.list_models(provider="voyageai")
        assert len(models) > 0
        assert all(m["provider"] == "voyageai" for m in models)


class TestClientContextManager:
    """Tests for context manager support."""

    def test_context_manager_sync(self):
        """Test synchronous context manager."""
        with Client() as client:
            assert client is not None

    @pytest.mark.asyncio
    async def test_context_manager_async(self):
        """Test async context manager."""
        async with Client() as client:
            assert client is not None
