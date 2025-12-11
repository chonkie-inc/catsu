"""Tests for Mixedbread AI provider."""

import os

import httpx
import pytest

from catsu.models import EmbedResponse
from catsu.providers.mixedbread import MixedbreadProvider
from catsu.utils.errors import InvalidInputError


class TestMixedbreadProvider:
    """Tests for Mixedbread AI provider."""

    @pytest.fixture
    def provider(self, mixedbread_api_key, catalog):
        """Create Mixedbread AI provider instance."""
        return MixedbreadProvider(
            http_client=httpx.Client(timeout=30),
            async_http_client=httpx.AsyncClient(timeout=30),
            catalog=catalog,
            api_key=mixedbread_api_key,
            max_retries=3,
            verbose=False,
        )

    def test_provider_initialization(self, provider):
        """Test that provider initializes correctly."""
        assert provider.PROVIDER_NAME == "mixedbread"
        assert provider.API_BASE_URL == "https://api.mixedbread.com/v1"
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

    def test_map_input_type_to_prompt(self, provider):
        """Test mapping generic input_type to Mixedbread prompt."""
        query_prompt = provider._map_input_type_to_prompt("query")
        assert (
            query_prompt == "Represent this sentence for searching relevant passages: "
        )
        assert provider._map_input_type_to_prompt("document") is None
        assert provider._map_input_type_to_prompt(None) is None

    def test_build_request_payload_basic(self, provider):
        """Test building basic API request payload."""
        payload = provider._build_request_payload(
            model="mxbai-embed-large-v1",
            inputs=["hello", "world"],
        )
        assert payload["model"] == "mxbai-embed-large-v1"
        assert payload["input"] == ["hello", "world"]
        assert payload["normalized"] is True
        assert payload["encoding_format"] == "float"

    def test_build_request_payload_with_input_type_query(self, provider):
        """Test building request payload with input_type=query."""
        payload = provider._build_request_payload(
            model="mxbai-embed-large-v1",
            inputs=["test query"],
            input_type="query",
        )
        assert (
            payload["prompt"]
            == "Represent this sentence for searching relevant passages: "
        )

    def test_build_request_payload_with_input_type_document(self, provider):
        """Test building request payload with input_type=document."""
        payload = provider._build_request_payload(
            model="mxbai-embed-large-v1",
            inputs=["test document"],
            input_type="document",
        )
        assert "prompt" not in payload

    def test_build_request_payload_with_custom_prompt(self, provider):
        """Test building request payload with custom prompt."""
        custom_prompt = "Custom task: "
        payload = provider._build_request_payload(
            model="mxbai-embed-large-v1",
            inputs=["test"],
            prompt=custom_prompt,
        )
        assert payload["prompt"] == custom_prompt

    def test_build_request_payload_with_dimensions(self, provider):
        """Test building request payload with dimensions."""
        payload = provider._build_request_payload(
            model="mxbai-embed-large-v1",
            inputs=["test"],
            dimensions=512,
        )
        assert payload["dimensions"] == 512

    def test_build_request_payload_with_normalized_false(self, provider):
        """Test building request payload with normalized=false."""
        payload = provider._build_request_payload(
            model="mxbai-embed-large-v1",
            inputs=["test"],
            normalized=False,
        )
        assert payload["normalized"] is False

    @pytest.mark.skipif(
        not os.getenv("MIXEDBREAD_API_KEY"),
        reason="Requires MIXEDBREAD_API_KEY environment variable",
    )
    def test_embed_single_text(self, skip_if_no_mixedbread_key, provider):
        """Test embedding a single text."""
        response = provider.embed(
            model="mxbai-embed-large-v1",
            inputs=["Hello, world!"],
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert len(response.embeddings[0]) == response.dimensions
        assert response.model == "mxbai-embed-large-v1"
        assert response.provider == "mixedbread"
        assert response.usage.tokens > 0
        assert response.usage.cost >= 0
        assert response.latency_ms > 0
        assert response.input_count == 1

    @pytest.mark.skipif(
        not os.getenv("MIXEDBREAD_API_KEY"),
        reason="Requires MIXEDBREAD_API_KEY environment variable",
    )
    def test_embed_with_query_input_type(self, skip_if_no_mixedbread_key, provider):
        """Test embedding with input_type=query."""
        response = provider.embed(
            model="mxbai-embed-large-v1",
            inputs=["What is machine learning?"],
            input_type="query",
        )

        assert isinstance(response, EmbedResponse)
        assert response.input_type == "query"

    @pytest.mark.skipif(
        not os.getenv("MIXEDBREAD_API_KEY"),
        reason="Requires MIXEDBREAD_API_KEY environment variable",
    )
    def test_embed_with_dimensions(self, skip_if_no_mixedbread_key, provider):
        """Test embedding with custom dimensions."""
        response = provider.embed(
            model="mxbai-embed-large-v1",
            inputs=["Test text"],
            dimensions=512,
        )

        assert isinstance(response, EmbedResponse)
        assert response.dimensions == 512
        assert len(response.embeddings[0]) == 512

    @pytest.mark.skipif(
        not os.getenv("MIXEDBREAD_API_KEY"),
        reason="Requires MIXEDBREAD_API_KEY environment variable",
    )
    @pytest.mark.asyncio
    async def test_aembed_single_text(self, skip_if_no_mixedbread_key, provider):
        """Test async embedding a single text."""
        response = await provider.aembed(
            model="mxbai-embed-large-v1",
            inputs=["Hello, async world!"],
        )

        assert isinstance(response, EmbedResponse)
        assert len(response.embeddings) == 1
        assert response.model == "mxbai-embed-large-v1"
        assert response.provider == "mixedbread"

    @pytest.mark.skipif(
        not os.getenv("MIXEDBREAD_API_KEY"),
        reason="Requires MIXEDBREAD_API_KEY environment variable",
    )
    @pytest.mark.asyncio
    async def test_aembed_with_dimensions(self, skip_if_no_mixedbread_key, provider):
        """Test async embedding with custom dimensions."""
        response = await provider.aembed(
            model="mxbai-embed-large-v1",
            inputs=["Async test"],
            dimensions=256,
        )

        assert isinstance(response, EmbedResponse)
        assert response.dimensions == 256
