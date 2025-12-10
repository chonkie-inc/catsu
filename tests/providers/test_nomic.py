"""Tests for Nomic provider."""

import os

import httpx
import pytest

from catsu.models import EmbedResponse
from catsu.providers.nomic import NomicProvider
from catsu.utils.errors import InvalidInputError


class TestNomicProvider:
    """Tests for Nomic provider."""

    @pytest.fixture
    def provider(self, nomic_api_key, catalog):
        """Create Nomic provider instance."""
        return NomicProvider(
            http_client=httpx.Client(timeout=30),
            async_http_client=httpx.AsyncClient(timeout=30),
            catalog=catalog,
            api_key=nomic_api_key,
            max_retries=3,
            verbose=False,
        )

    def test_provider_initialization(self, provider):
        """Test that provider initializes correctly."""
        assert provider.PROVIDER_NAME == "nomic"
        assert provider.API_BASE_URL == "https://api-atlas.nomic.ai/v1"
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

    def test_map_input_type_to_task_type(self, provider):
        """Test mapping generic input_type to Nomic task_type."""
        assert provider._map_input_type_to_task_type("query") == "search_query"
        assert provider._map_input_type_to_task_type("document") == "search_document"
        assert provider._map_input_type_to_task_type(None) == "search_document"
        assert provider._map_input_type_to_task_type("other") == "search_document"

    def test_build_request_payload_basic(self, provider):
        """Test building basic API request payload."""
        payload = provider._build_request_payload(
            model="nomic-embed-text-v1.5",
            inputs=["hello", "world"],
        )
        assert payload["model"] == "nomic-embed-text-v1.5"
        assert payload["texts"] == ["hello", "world"]
        assert payload["long_text_mode"] == "mean"
        assert payload["max_tokens_per_text"] == 8192

    def test_build_request_payload_with_task_type(self, provider):
        """Test building request payload with task_type."""
        payload = provider._build_request_payload(
            model="nomic-embed-text-v1.5",
            inputs=["test"],
            task_type="search_query",
        )
        assert payload["task_type"] == "search_query"

    def test_build_request_payload_invalid_task_type(self, provider):
        """Test that invalid task_type raises error."""
        with pytest.raises(InvalidInputError):
            provider._build_request_payload(
                model="nomic-embed-text-v1.5",
                inputs=["test"],
                task_type="invalid_task",
            )

    def test_build_request_payload_with_dimensions_v15(self, provider):
        """Test building request payload with dimensions for v1.5."""
        payload = provider._build_request_payload(
            model="nomic-embed-text-v1.5",
            inputs=["test"],
            dimensions=256,
        )
        assert payload["dimensionality"] == 256

    def test_build_request_payload_dimensions_wrong_model(self, provider):
        """Test that dimensions on v1 raises error."""
        with pytest.raises(InvalidInputError):
            provider._build_request_payload(
                model="nomic-embed-text-v1",
                inputs=["test"],
                dimensions=256,
            )

    def test_build_request_payload_dimensions_out_of_range(self, provider):
        """Test that dimensions out of range raises error."""
        with pytest.raises(InvalidInputError):
            provider._build_request_payload(
                model="nomic-embed-text-v1.5",
                inputs=["test"],
                dimensions=1024,  # Max is 768
            )

        with pytest.raises(InvalidInputError):
            provider._build_request_payload(
                model="nomic-embed-text-v1.5",
                inputs=["test"],
                dimensions=32,  # Min is 64
            )

    @pytest.mark.skipif(
        not os.getenv("NOMIC_API_KEY"),
        reason="Requires NOMIC_API_KEY environment variable",
    )
    def test_embed_single_text_v15(self, skip_if_no_nomic_key, provider):
        """Test embedding a single text with v1.5."""
        response = provider.embed(
            model="nomic-embed-text-v1.5",
            inputs=["Hello, world!"],
            task_type="search_query",
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert len(response.embeddings[0]) == response.dimensions
        assert response.model == "nomic-embed-text-v1.5"
        assert response.provider == "nomic"
        assert response.usage.tokens > 0
        assert response.usage.cost >= 0
        assert response.latency_ms > 0
        assert response.input_count == 1

    @pytest.mark.skipif(
        not os.getenv("NOMIC_API_KEY"),
        reason="Requires NOMIC_API_KEY environment variable",
    )
    def test_embed_multiple_texts_v15(self, skip_if_no_nomic_key, provider):
        """Test embedding multiple texts with v1.5."""
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is transforming technology",
            "Python is a versatile programming language",
        ]
        response = provider.embed(
            model="nomic-embed-text-v1.5",
            inputs=texts,
            task_type="search_document",
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 3
        assert response.input_count == 3
        assert all(len(emb) == response.dimensions for emb in response.embeddings)

    @pytest.mark.skipif(
        not os.getenv("NOMIC_API_KEY"),
        reason="Requires NOMIC_API_KEY environment variable",
    )
    def test_embed_with_dimensions_v15(self, skip_if_no_nomic_key, provider):
        """Test embedding with custom dimensions for v1.5."""
        response = provider.embed(
            model="nomic-embed-text-v1.5",
            inputs=["Test text"],
            task_type="clustering",
            dimensions=256,
        )

        assert isinstance(response, EmbedResponse)
        assert response.dimensions == 256
        assert len(response.embeddings[0]) == 256

    @pytest.mark.skipif(
        not os.getenv("NOMIC_API_KEY"),
        reason="Requires NOMIC_API_KEY environment variable",
    )
    def test_embed_v1(self, skip_if_no_nomic_key, provider):
        """Test embedding with v1 model."""
        response = provider.embed(
            model="nomic-embed-text-v1",
            inputs=["Hello, world!"],
            task_type="classification",
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert response.model == "nomic-embed-text-v1"
        assert response.dimensions == 768  # v1 has fixed 768 dimensions

    @pytest.mark.skipif(
        not os.getenv("NOMIC_API_KEY"),
        reason="Requires NOMIC_API_KEY environment variable",
    )
    def test_embed_with_input_type_mapping(self, skip_if_no_nomic_key, provider):
        """Test that input_type is correctly mapped to task_type."""
        response = provider.embed(
            model="nomic-embed-text-v1.5",
            inputs=["Test query"],
            input_type="query",  # Should map to search_query
        )

        assert isinstance(response, EmbedResponse)
        assert response.input_type == "query"

    @pytest.mark.skipif(
        not os.getenv("NOMIC_API_KEY"),
        reason="Requires NOMIC_API_KEY environment variable",
    )
    @pytest.mark.asyncio
    async def test_aembed_single_text_v15(self, skip_if_no_nomic_key, provider):
        """Test async embedding a single text with v1.5."""
        response = await provider.aembed(
            model="nomic-embed-text-v1.5",
            inputs=["Hello, async world!"],
            task_type="search_query",
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert response.model == "nomic-embed-text-v1.5"
        assert response.provider == "nomic"

    @pytest.mark.skipif(
        not os.getenv("NOMIC_API_KEY"),
        reason="Requires NOMIC_API_KEY environment variable",
    )
    @pytest.mark.asyncio
    async def test_aembed_with_dimensions_v15(self, skip_if_no_nomic_key, provider):
        """Test async embedding with custom dimensions."""
        response = await provider.aembed(
            model="nomic-embed-text-v1.5",
            inputs=["Async test"],
            dimensions=128,
        )

        assert isinstance(response, EmbedResponse)
        assert response.dimensions == 128
