"""Tests for custom exceptions."""

from mimie.utils.errors import (
    AmbiguousModelError,
    AuthenticationError,
    InvalidInputError,
    MimieError,
    ModelNotFoundError,
    NetworkError,
    ProviderError,
    RateLimitError,
    TimeoutError,
)


class TestMimieError:
    """Tests for base MimieError."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = MimieError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.details == {}

    def test_error_with_details(self):
        """Test error with additional details."""
        error = MimieError("Error occurred", details={"key": "value"})
        assert "key=value" in str(error)
        assert error.details["key"] == "value"


class TestModelNotFoundError:
    """Tests for ModelNotFoundError."""

    def test_model_not_found_without_provider(self):
        """Test model not found without provider."""
        error = ModelNotFoundError(model="test-model")
        assert "test-model" in str(error)
        assert "not found in any provider" in str(error)
        assert error.model == "test-model"
        assert error.provider is None

    def test_model_not_found_with_provider(self):
        """Test model not found with provider."""
        error = ModelNotFoundError(model="test-model", provider="voyageai")
        assert "test-model" in str(error)
        assert "voyageai" in str(error)
        assert error.model == "test-model"
        assert error.provider == "voyageai"


class TestAmbiguousModelError:
    """Tests for AmbiguousModelError."""

    def test_ambiguous_model(self):
        """Test ambiguous model error."""
        error = AmbiguousModelError(model="text-embed", providers=["openai", "cohere"])
        assert "text-embed" in str(error)
        assert "openai" in str(error)
        assert "cohere" in str(error)
        assert error.model == "text-embed"
        assert error.providers == ["openai", "cohere"]


class TestProviderError:
    """Tests for ProviderError."""

    def test_provider_error_basic(self):
        """Test basic provider error."""
        error = ProviderError("API failed")
        assert str(error) == "API failed"

    def test_provider_error_with_status(self):
        """Test provider error with status code."""
        error = ProviderError(
            message="API failed", provider="voyageai", status_code=500
        )
        assert "voyageai" in str(error)
        assert error.status_code == 500


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_rate_limit_error(self):
        """Test rate limit error."""
        error = RateLimitError(
            message="Rate limited", provider="voyageai", retry_after=60
        )
        assert error.status_code == 429
        assert error.retry_after == 60
        assert "60s" in str(error)


class TestAuthenticationError:
    """Tests for AuthenticationError."""

    def test_authentication_error(self):
        """Test authentication error."""
        error = AuthenticationError(message="Invalid API key", provider="voyageai")
        assert error.status_code == 401
        assert "Invalid API key" in str(error)


class TestInvalidInputError:
    """Tests for InvalidInputError."""

    def test_invalid_input_error(self):
        """Test invalid input error."""
        error = InvalidInputError(message="Input cannot be empty", parameter="input")
        assert "Input cannot be empty" in str(error)
        assert error.parameter == "input"


class TestTimeoutError:
    """Tests for TimeoutError."""

    def test_timeout_error(self):
        """Test timeout error."""
        error = TimeoutError(
            message="Request timed out", provider="voyageai", timeout=30.0
        )
        assert error.status_code == 408
        assert error.timeout == 30.0
        assert "30.0s" in str(error)


class TestNetworkError:
    """Tests for NetworkError."""

    def test_network_error(self):
        """Test network error."""
        error = NetworkError(message="Connection refused", provider="voyageai")
        assert "Connection refused" in str(error)
        assert error.provider == "voyageai"
