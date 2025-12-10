"""Shared pytest fixtures for Catsu tests."""

import os

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require API keys",
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers",
        "requires_api_key: mark test as requiring an API key (skips if not set)",
    )


@pytest.fixture
def voyage_api_key():
    """Get VoyageAI API key from environment."""
    return os.getenv("VOYAGE_API_KEY")


@pytest.fixture
def has_voyage_api_key():
    """Check if VoyageAI API key is available."""
    return os.getenv("VOYAGE_API_KEY") is not None


@pytest.fixture
def skip_if_no_voyage_key():
    """Skip test if VoyageAI API key is not set."""
    if not os.getenv("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY not set")


@pytest.fixture
def cohere_api_key():
    """Get Cohere API key from environment."""
    return os.getenv("COHERE_API_KEY")


@pytest.fixture
def has_cohere_api_key():
    """Check if Cohere API key is available."""
    return os.getenv("COHERE_API_KEY") is not None


@pytest.fixture
def skip_if_no_cohere_key():
    """Skip test if Cohere API key is not set."""
    if not os.getenv("COHERE_API_KEY"):
        pytest.skip("COHERE_API_KEY not set")


@pytest.fixture
def jina_api_key():
    """Get Jina AI API key from environment."""
    return os.getenv("JINA_API_KEY")


@pytest.fixture
def has_jina_api_key():
    """Check if Jina AI API key is available."""
    return os.getenv("JINA_API_KEY") is not None


@pytest.fixture
def skip_if_no_jina_key():
    """Skip test if Jina AI API key is not set."""
    if not os.getenv("JINA_API_KEY"):
        pytest.skip("JINA_API_KEY not set")


@pytest.fixture
def gemini_api_key():
    """Get Gemini API key from environment."""
    return os.getenv("GEMINI_API_KEY")


@pytest.fixture
def has_gemini_api_key():
    """Check if Gemini API key is available."""
    return os.getenv("GEMINI_API_KEY") is not None


@pytest.fixture
def skip_if_no_gemini_key():
    """Skip test if Gemini API key is not set."""
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")


@pytest.fixture
def mistral_api_key():
    """Get Mistral API key from environment."""
    return os.getenv("MISTRAL_API_KEY")


@pytest.fixture
def has_mistral_api_key():
    """Check if Mistral API key is available."""
    return os.getenv("MISTRAL_API_KEY") is not None


@pytest.fixture
def skip_if_no_mistral_key():
    """Skip test if Mistral API key is not set."""
    if not os.getenv("MISTRAL_API_KEY"):
        pytest.skip("MISTRAL_API_KEY not set")


@pytest.fixture
def nomic_api_key():
    """Get Nomic API key from environment."""
    return os.getenv("NOMIC_API_KEY")


@pytest.fixture
def has_nomic_api_key():
    """Check if Nomic API key is available."""
    return os.getenv("NOMIC_API_KEY") is not None


@pytest.fixture
def skip_if_no_nomic_key():
    """Skip test if Nomic API key is not set."""
    if not os.getenv("NOMIC_API_KEY"):
        pytest.skip("NOMIC_API_KEY not set")
