"""Tests for ModelCatalog."""

import pytest

from mimie.catalog import ModelCatalog
from mimie.models import ModelInfo
from mimie.utils.errors import ModelNotFoundError


class TestModelCatalog:
    """Tests for ModelCatalog class."""

    @pytest.fixture
    def catalog(self):
        """Create a ModelCatalog instance."""
        return ModelCatalog()

    def test_catalog_initialization(self, catalog):
        """Test that catalog initializes successfully."""
        assert catalog is not None
        assert len(catalog._models) > 0

    def test_get_model_info(self, catalog):
        """Test getting model info for valid model."""
        info = catalog.get_model_info("voyageai", "voyage-3")
        assert isinstance(info, ModelInfo)
        assert info.name == "voyage-3"
        assert info.provider == "voyageai"
        assert info.dimensions == 1024
        assert info.cost_per_million_tokens == 0.06

    def test_get_model_info_not_found(self, catalog):
        """Test getting model info for non-existent model."""
        with pytest.raises(ModelNotFoundError) as exc_info:
            catalog.get_model_info("voyageai", "nonexistent")
        assert exc_info.value.model == "nonexistent"
        assert exc_info.value.provider == "voyageai"

    def test_get_model_info_provider_not_found(self, catalog):
        """Test getting model info for non-existent provider."""
        with pytest.raises(ModelNotFoundError):
            catalog.get_model_info("nonexistent-provider", "some-model")

    def test_list_models_all(self, catalog):
        """Test listing all models."""
        models = catalog.list_models()
        assert len(models) > 0
        assert all(isinstance(m, ModelInfo) for m in models)

    def test_list_models_by_provider(self, catalog):
        """Test listing models filtered by provider."""
        models = catalog.list_models(provider="voyageai")
        assert len(models) > 0
        assert all(m.provider == "voyageai" for m in models)

    def test_list_models_unknown_provider(self, catalog):
        """Test listing models for unknown provider returns empty list."""
        models = catalog.list_models(provider="unknown")
        assert models == []

    def test_auto_detect_provider(self, catalog):
        """Test auto-detecting provider for unique model name."""
        provider = catalog.auto_detect_provider("voyage-3")
        assert provider == "voyageai"

    def test_auto_detect_provider_not_found(self, catalog):
        """Test auto-detection returns None for non-existent model."""
        provider = catalog.auto_detect_provider("nonexistent-model")
        assert provider is None

    def test_list_providers(self, catalog):
        """Test listing all providers."""
        providers = catalog.list_providers()
        assert "voyageai" in providers
        assert len(providers) > 0

    def test_has_model(self, catalog):
        """Test checking if model exists."""
        assert catalog.has_model("voyageai", "voyage-3") is True
        assert catalog.has_model("voyageai", "nonexistent") is False
        assert catalog.has_model("unknown", "voyage-3") is False

    def test_get_models_by_capability(self, catalog):
        """Test filtering models by capabilities."""
        # Test filtering by dimension support
        models = catalog.get_models_by_capability(supports_dimensions=True)
        assert len(models) > 0
        assert all(m.supports_dimensions for m in models)

        # Test filtering by cost
        cheap_models = catalog.get_models_by_capability(max_cost=0.10)
        assert all(m.cost_per_million_tokens <= 0.10 for m in cheap_models)

    def test_catalog_repr(self, catalog):
        """Test catalog string representation."""
        repr_str = repr(catalog)
        assert "ModelCatalog" in repr_str
        assert "voyageai" in repr_str

    def test_voyage_models_loaded(self, catalog):
        """Test that all expected VoyageAI models are loaded."""
        models = catalog.list_models(provider="voyageai")
        model_names = [m.name for m in models]

        # Check key models exist
        assert "voyage-3" in model_names
        assert "voyage-3-lite" in model_names
        assert "voyage-3.5" in model_names
        assert "voyage-code-3" in model_names

    def test_model_tokenizer_config(self, catalog):
        """Test that models have tokenizer configuration."""
        info = catalog.get_model_info("voyageai", "voyage-3")
        assert info.tokenizer is not None
        assert "repo" in info.tokenizer
        assert info.tokenizer["repo"] == "voyageai/voyage-3"
