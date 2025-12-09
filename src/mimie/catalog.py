"""Model catalog for Mimie.

Manages model metadata, provides model discovery, and enables auto-detection
of providers based on model names.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from .models import ModelInfo
from .utils.errors import AmbiguousModelError, ModelNotFoundError


class ModelCatalog:
    """Central catalog for embedding model metadata.

    Loads model information from the bundled models.json file and provides
    methods for querying model capabilities, pricing, and specifications.

    The catalog supports:
    - Looking up models by name and provider
    - Auto-detecting providers from model names
    - Listing available models with optional filtering

    Example:
        >>> catalog = ModelCatalog()
        >>> model_info = catalog.get_model_info("voyageai", "voyage-3")
        >>> print(model_info.dimensions)
        1024
        >>> print(model_info.cost_per_million_tokens)
        0.06

    """

    def __init__(self, models_path: Optional[Path] = None) -> None:
        """Initialize the model catalog.

        Args:
            models_path: Optional custom path to models.json file.
                        If not provided, uses bundled models.json.

        """
        if models_path is None:
            # Use bundled models.json
            models_path = Path(__file__).parent / "data" / "models.json"

        self._models_path = models_path
        self._models: Dict[str, Dict[str, ModelInfo]] = {}
        self._load_models()

    def _load_models(self) -> None:
        """Load models from JSON file into memory."""
        if not self._models_path.exists():
            raise FileNotFoundError(
                f"Models file not found: {self._models_path}"
            )

        with open(self._models_path, "r") as f:
            data = json.load(f)

        # Parse JSON into ModelInfo objects
        # Structure: {"provider": [{"name": ..., ...}, ...]}
        for provider, models_list in data.items():
            self._models[provider] = {}
            for model_data in models_list:
                model_info = ModelInfo(**model_data)
                self._models[provider][model_info.name] = model_info

    def get_model_info(self, provider: str, model: str) -> ModelInfo:
        """Get model information for a specific provider and model.

        Args:
            provider: Provider name (e.g., "voyageai")
            model: Model name (e.g., "voyage-3")

        Returns:
            ModelInfo object with model specifications

        Raises:
            ModelNotFoundError: If provider or model not found

        Example:
            >>> catalog = ModelCatalog()
            >>> info = catalog.get_model_info("voyageai", "voyage-3")
            >>> print(f"{info.dimensions} dimensions, ${info.cost_per_million_tokens}/M")
            1024 dimensions, $0.06/M

        """
        if provider not in self._models:
            available = list(self._models.keys())
            raise ModelNotFoundError(
                model=model,
                provider=provider,
                details={"available_providers": available},
            )

        if model not in self._models[provider]:
            available = list(self._models[provider].keys())
            raise ModelNotFoundError(
                model=model,
                provider=provider,
                details={"available_models": available},
            )

        return self._models[provider][model]

    def list_models(self, provider: Optional[str] = None) -> List[ModelInfo]:
        """List available models, optionally filtered by provider.

        Args:
            provider: Optional provider name to filter by.
                     If None, returns all models from all providers.

        Returns:
            List of ModelInfo objects

        Example:
            >>> catalog = ModelCatalog()
            >>> # List all models
            >>> all_models = catalog.list_models()
            >>> # List only VoyageAI models
            >>> voyage_models = catalog.list_models(provider="voyageai")
            >>> for model in voyage_models:
            ...     print(f"{model.name}: {model.dimensions} dims")

        """
        if provider is not None:
            if provider not in self._models:
                return []
            return list(self._models[provider].values())

        # Return all models from all providers
        all_models = []
        for provider_models in self._models.values():
            all_models.extend(provider_models.values())
        return all_models

    def auto_detect_provider(self, model: str) -> Optional[str]:
        """Attempt to auto-detect the provider for a given model name.

        Searches all providers to find which one(s) have this model.
        If exactly one provider has the model, returns that provider.
        If multiple providers have the model, raises AmbiguousModelError.
        If no provider has the model, returns None.

        Args:
            model: Model name to search for

        Returns:
            Provider name if found uniquely, None if not found

        Raises:
            AmbiguousModelError: If model exists in multiple providers

        Example:
            >>> catalog = ModelCatalog()
            >>> provider = catalog.auto_detect_provider("voyage-3")
            >>> print(provider)
            voyageai

        """
        matching_providers = []

        for provider, models in self._models.items():
            if model in models:
                matching_providers.append(provider)

        if len(matching_providers) == 0:
            return None
        elif len(matching_providers) == 1:
            return matching_providers[0]
        else:
            raise AmbiguousModelError(
                model=model,
                providers=matching_providers,
            )

    def list_providers(self) -> List[str]:
        """List all available providers.

        Returns:
            List of provider names

        Example:
            >>> catalog = ModelCatalog()
            >>> providers = catalog.list_providers()
            >>> print(providers)
            ['voyageai']

        """
        return list(self._models.keys())

    def has_model(self, provider: str, model: str) -> bool:
        """Check if a specific model exists for a provider.

        Args:
            provider: Provider name
            model: Model name

        Returns:
            True if model exists, False otherwise

        Example:
            >>> catalog = ModelCatalog()
            >>> catalog.has_model("voyageai", "voyage-3")
            True
            >>> catalog.has_model("voyageai", "nonexistent")
            False

        """
        return (
            provider in self._models
            and model in self._models[provider]
        )

    def get_models_by_capability(
        self,
        supports_batching: Optional[bool] = None,
        supports_input_type: Optional[bool] = None,
        supports_dimensions: Optional[bool] = None,
        max_cost: Optional[float] = None,
    ) -> List[ModelInfo]:
        """Filter models by capabilities and constraints.

        Args:
            supports_batching: Filter by batching support
            supports_input_type: Filter by input_type parameter support
            supports_dimensions: Filter by custom dimensions support
            max_cost: Maximum cost per million tokens

        Returns:
            List of ModelInfo objects matching the criteria

        Example:
            >>> catalog = ModelCatalog()
            >>> # Find cheap models with dimension support
            >>> cheap_models = catalog.get_models_by_capability(
            ...     supports_dimensions=True,
            ...     max_cost=0.10
            ... )

        """
        results = []

        for models in self._models.values():
            for model_info in models.values():
                # Check all filters
                if supports_batching is not None:
                    if model_info.supports_batching != supports_batching:
                        continue

                if supports_input_type is not None:
                    if model_info.supports_input_type != supports_input_type:
                        continue

                if supports_dimensions is not None:
                    if model_info.supports_dimensions != supports_dimensions:
                        continue

                if max_cost is not None:
                    if model_info.cost_per_million_tokens > max_cost:
                        continue

                results.append(model_info)

        return results

    def __repr__(self) -> str:
        """Return string representation of the catalog."""
        total_models = sum(len(models) for models in self._models.values())
        providers = list(self._models.keys())
        return (
            f"ModelCatalog(providers={providers}, "
            f"total_models={total_models})"
        )
