"""Shared pytest fixtures for Mimie tests."""

import os

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require API keys"
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers",
        "requires_api_key: mark test as requiring an API key (skips if not set)"
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
