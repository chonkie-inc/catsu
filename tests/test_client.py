"""Tests for Client class."""

import os
from unittest.mock import MagicMock, patch

import pytest

from catsu import Client
from catsu.models import EmbedResponse, Usage
from catsu.utils.errors import (
    ConfigurationError,
    FallbackExhaustedError,
    InvalidInputError,
    ModelNotFoundError,
    NetworkError,
    RateLimitError,
    UnsupportedFeatureError,
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

    def test_embed_unsupported_dimensions(self):
        """Test that using dimensions on unsupported model raises error."""
        client = Client()
        # text-embedding-ada-002 does not support custom dimensions
        with pytest.raises(UnsupportedFeatureError) as exc_info:
            client.embed(
                model="openai:text-embedding-ada-002",
                input="test",
                dimensions=512,
            )
        assert "does not support custom dimensions" in str(exc_info.value)
        assert exc_info.value.model == "text-embedding-ada-002"
        assert exc_info.value.provider == "openai"
        assert exc_info.value.feature == "dimensions"

    @pytest.mark.skipif(
        not os.getenv("VOYAGE_API_KEY"),
        reason="Requires VOYAGE_API_KEY environment variable",
    )
    def test_embed_with_dimensions_supported(self, skip_if_no_voyage_key):
        """Test that dimensions works on supported model."""
        client = Client()
        response = client.embed(
            model="voyage-3-lite",
            input="Test with dimensions",
            dimensions=512,
        )
        assert isinstance(response, EmbedResponse)
        assert response.dimensions == 512


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


class TestFallbackNormalization:
    """Tests for fallback normalization."""

    def test_normalize_fallbacks_none(self):
        """Test normalizing None returns empty list."""
        client = Client()
        result = client._normalize_fallbacks(None)
        assert result == []

    def test_normalize_fallbacks_string(self):
        """Test normalizing string returns single-item list."""
        client = Client()
        result = client._normalize_fallbacks("text-embedding-3-small")
        assert result == [{"model": "text-embedding-3-small"}]

    def test_normalize_fallbacks_dict(self):
        """Test normalizing dict returns single-item list."""
        client = Client()
        config = {"model": "text-embedding-3-small", "api_key": "test-key"}
        result = client._normalize_fallbacks(config)
        assert result == [config]

    def test_normalize_fallbacks_list_of_strings(self):
        """Test normalizing list of strings."""
        client = Client()
        result = client._normalize_fallbacks(["model-a", "model-b"])
        assert result == [{"model": "model-a"}, {"model": "model-b"}]

    def test_normalize_fallbacks_mixed_list(self):
        """Test normalizing mixed list of strings and dicts."""
        client = Client()
        fallbacks = [
            "model-a",
            {"model": "model-b", "api_key": "key-b"},
        ]
        result = client._normalize_fallbacks(fallbacks)
        assert result == [
            {"model": "model-a"},
            {"model": "model-b", "api_key": "key-b"},
        ]


class TestFallbackCredentialValidation:
    """Tests for fallback credential validation."""

    def test_validate_credentials_missing_primary(self, monkeypatch):
        """Test ConfigurationError raised when primary model API key missing."""
        # Clear all API keys
        for key in ["VOYAGE_API_KEY", "OPENAI_API_KEY", "COHERE_API_KEY"]:
            monkeypatch.delenv(key, raising=False)

        client = Client()
        with pytest.raises(ConfigurationError) as exc_info:
            client._validate_fallback_credentials(
                "voyageai:voyage-3",
                [{"model": "openai:text-embedding-3-small"}],
            )
        assert "VOYAGE_API_KEY" in str(exc_info.value)

    def test_validate_credentials_missing_fallback(self, monkeypatch):
        """Test ConfigurationError raised when fallback API key missing."""
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        client = Client()
        with pytest.raises(ConfigurationError) as exc_info:
            client._validate_fallback_credentials(
                "voyageai:voyage-3",
                [{"model": "openai:text-embedding-3-small"}],
            )
        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_validate_credentials_all_present(self, monkeypatch):
        """Test no error when all API keys are present."""
        monkeypatch.setenv("VOYAGE_API_KEY", "key-1")
        monkeypatch.setenv("OPENAI_API_KEY", "key-2")

        client = Client()
        # Should not raise
        client._validate_fallback_credentials(
            "voyageai:voyage-3",
            [{"model": "openai:text-embedding-3-small"}],
        )


class TestFallbackDimensionFiltering:
    """Tests for fallback dimension filtering."""

    def test_can_produce_dimensions_exact_match(self):
        """Test model with exact matching dimensions is compatible."""
        client = Client()
        # embed-english-v3.0 has 1024 dims and doesn't support custom
        assert client._can_produce_dimensions("cohere", "embed-english-v3.0", 1024)

    def test_can_produce_dimensions_supports_custom(self):
        """Test model with custom dimension support is compatible."""
        client = Client()
        # voyage-3 supports custom dimensions
        assert client._can_produce_dimensions("voyageai", "voyage-3", 512)

    def test_can_produce_dimensions_mismatch(self):
        """Test model with mismatched dimensions is not compatible."""
        client = Client()
        # text-embedding-ada-002 has 1536 dims and doesn't support custom
        assert not client._can_produce_dimensions("openai", "text-embedding-ada-002", 1024)

    def test_filter_compatible_fallbacks(self):
        """Test filtering fallbacks by dimension compatibility."""
        client = Client()
        fallbacks = [
            {"model": "voyageai:voyage-3-lite"},  # Supports custom dims
            {"model": "openai:text-embedding-ada-002"},  # 1536 fixed
        ]
        compatible = client._filter_compatible_fallbacks(1024, fallbacks)
        # Only voyage-3-lite should be compatible (supports custom dims)
        assert len(compatible) == 1
        assert compatible[0]["model"] == "voyageai:voyage-3-lite"


class TestFallbackErrorClassification:
    """Tests for fallback error classification."""

    def test_rate_limit_error_triggers_fallback(self):
        """Test RateLimitError triggers fallback."""
        client = Client()
        error = RateLimitError("Rate limited", provider="test")
        assert client._is_fallback_error(error)

    def test_network_error_triggers_fallback(self):
        """Test NetworkError triggers fallback."""
        client = Client()
        error = NetworkError("Connection failed", provider="test")
        assert client._is_fallback_error(error)

    def test_invalid_input_error_does_not_trigger_fallback(self):
        """Test InvalidInputError does not trigger fallback."""
        client = Client()
        error = InvalidInputError("Bad input")
        assert not client._is_fallback_error(error)


class TestFallbackClientConfiguration:
    """Tests for client-level fallback configuration."""

    def test_client_fallback_defaults(self):
        """Test client initializes with fallback defaults."""
        client = Client()
        assert client.fallbacks is None
        assert client.allow_unsafe_fallback is False
        assert client.base_delay == 1.0
        assert client.max_delay == 10.0

    def test_client_fallback_custom_config(self):
        """Test client with custom fallback config."""
        client = Client(
            fallbacks=["text-embedding-3-small"],
            allow_unsafe_fallback=True,
            base_delay=2.0,
            max_delay=20.0,
        )
        assert client.fallbacks == ["text-embedding-3-small"]
        assert client.allow_unsafe_fallback is True
        assert client.base_delay == 2.0
        assert client.max_delay == 20.0


class TestGetCompatibleFallbacks:
    """Tests for get_compatible_fallbacks helper."""

    def test_get_compatible_fallbacks_returns_list(self):
        """Test get_compatible_fallbacks returns a list."""
        client = Client()
        fallbacks = client.get_compatible_fallbacks("voyage-3")
        assert isinstance(fallbacks, list)

    def test_get_compatible_fallbacks_excludes_same_model(self):
        """Test get_compatible_fallbacks excludes the input model."""
        client = Client()
        fallbacks = client.get_compatible_fallbacks("voyage-3")
        assert "voyageai:voyage-3" not in fallbacks


class TestFallbackExhaustedError:
    """Tests for FallbackExhaustedError."""

    def test_fallback_exhausted_error_attributes(self):
        """Test FallbackExhaustedError has correct attributes."""
        error = FallbackExhaustedError(
            primary_model="voyage-3",
            fallbacks=["model-a", "model-b"],
            errors={"voyage-3": RateLimitError("Rate limited")},
            cycles_attempted=3,
        )
        assert error.primary_model == "voyage-3"
        assert error.fallbacks == ["model-a", "model-b"]
        assert "voyage-3" in error.errors
        assert error.cycles_attempted == 3
        assert "3 cycle(s)" in str(error)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_configuration_error_message(self):
        """Test ConfigurationError has correct message."""
        error = ConfigurationError("Missing API key")
        assert "Missing API key" in str(error)


class TestEmbedResponseFallbackMetadata:
    """Tests for EmbedResponse fallback metadata."""

    def test_embed_response_default_fallback_fields(self):
        """Test EmbedResponse has default fallback field values."""
        response = EmbedResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            model="voyage-3",
            provider="voyageai",
            dimensions=3,
            usage=Usage(tokens=10, cost=0.000001),
            latency_ms=100.0,
            input_count=1,
        )
        assert response.fallback_used is False
        assert response.requested_model is None
        assert response.fallback_reason is None

    def test_embed_response_with_fallback_metadata(self):
        """Test EmbedResponse with fallback metadata."""
        response = EmbedResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            model="text-embedding-3-small",
            provider="openai",
            dimensions=3,
            usage=Usage(tokens=10, cost=0.000001),
            latency_ms=100.0,
            input_count=1,
            fallback_used=True,
            requested_model="voyage-3",
            fallback_reason="RateLimitError",
        )
        assert response.fallback_used is True
        assert response.requested_model == "voyage-3"
        assert response.fallback_reason == "RateLimitError"

    def test_embed_response_repr_with_fallback(self):
        """Test EmbedResponse repr includes fallback info when used."""
        response = EmbedResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            model="text-embedding-3-small",
            provider="openai",
            dimensions=3,
            usage=Usage(tokens=10, cost=0.000001),
            latency_ms=100.0,
            input_count=1,
            fallback_used=True,
            requested_model="voyage-3",
        )
        repr_str = repr(response)
        assert "fallback_used=True" in repr_str
        assert "requested_model='voyage-3'" in repr_str
