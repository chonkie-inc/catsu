"""Custom exception classes for Mimie.

Provides a hierarchy of exceptions for different error scenarios, making it
easier to handle and debug issues when working with embedding providers.
"""

from typing import Any, Dict, Optional


class MimieError(Exception):
    """Base exception for all Mimie errors.

    All custom exceptions in Mimie inherit from this class, making it easy
    to catch any Mimie-related error.

    Attributes:
        message: Error message
        details: Optional additional details about the error

    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MimieError.

        Args:
            message: Error message
            details: Optional dictionary with additional error details

        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """String representation of the error."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ProviderError(MimieError):
    """Exception raised for provider-specific errors.

    Raised when an embedding provider returns an error or fails to process
    a request.

    Attributes:
        provider: Name of the provider that raised the error
        status_code: HTTP status code if applicable

    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize ProviderError.

        Args:
            message: Error message
            provider: Provider name (e.g., "voyageai")
            status_code: HTTP status code if applicable
            details: Optional additional error details

        """
        self.provider = provider
        self.status_code = status_code
        details = details or {}
        if provider:
            details["provider"] = provider
        if status_code:
            details["status_code"] = status_code
        super().__init__(message, details)


class ModelNotFoundError(MimieError):
    """Exception raised when a requested model is not found.

    Raised when attempting to use a model that doesn't exist in any provider
    or in the specified provider.

    Example:
        >>> client.embed(model="nonexistent-model", input="test")
        ModelNotFoundError: Model 'nonexistent-model' not found

    """

    def __init__(
        self,
        model: str,
        provider: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize ModelNotFoundError.

        Args:
            model: Model name that was not found
            provider: Provider name if specified
            details: Optional additional error details

        """
        self.model = model
        self.provider = provider

        if provider:
            message = f"Model '{model}' not found in provider '{provider}'"
        else:
            message = f"Model '{model}' not found in any provider"

        details = details or {}
        details["model"] = model
        if provider:
            details["provider"] = provider

        super().__init__(message, details)


class AmbiguousModelError(MimieError):
    """Exception raised when a model name is ambiguous.

    Raised when attempting to auto-detect a provider but the model name
    exists in multiple providers.

    Example:
        >>> # If "text-embed" exists in both OpenAI and Cohere
        >>> client.embed(model="text-embed", input="test")
        AmbiguousModelError: Model 'text-embed' found in multiple providers

    """

    def __init__(
        self,
        model: str,
        providers: list[str],
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize AmbiguousModelError.

        Args:
            model: Ambiguous model name
            providers: List of providers that have this model
            details: Optional additional error details

        """
        self.model = model
        self.providers = providers

        providers_str = ", ".join(providers)
        message = (
            f"Model '{model}' found in multiple providers: {providers_str}. "
            f"Please specify the provider explicitly."
        )

        details = details or {}
        details["model"] = model
        details["providers"] = providers

        super().__init__(message, details)


class RateLimitError(ProviderError):
    """Exception raised when a rate limit is exceeded.

    Raised when a provider returns a rate limit error (typically HTTP 429).

    Attributes:
        retry_after: Seconds to wait before retrying (if provided by API)

    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize RateLimitError.

        Args:
            message: Error message
            provider: Provider name
            retry_after: Seconds to wait before retrying
            details: Optional additional error details

        """
        self.retry_after = retry_after

        details = details or {}
        if retry_after:
            details["retry_after"] = retry_after
            message = f"{message} (retry after {retry_after}s)"

        super().__init__(
            message=message,
            provider=provider,
            status_code=429,
            details=details,
        )


class AuthenticationError(ProviderError):
    """Exception raised for authentication failures.

    Raised when API credentials are missing, invalid, or expired.

    Example:
        >>> client.embed(model="voyage-3", input="test")
        AuthenticationError: Invalid API key for provider 'voyageai'

    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize AuthenticationError.

        Args:
            message: Error message
            provider: Provider name
            details: Optional additional error details

        """
        super().__init__(
            message=message,
            provider=provider,
            status_code=401,
            details=details,
        )


class InvalidInputError(MimieError):
    """Exception raised for invalid input parameters.

    Raised when the provided input doesn't meet the requirements (e.g.,
    empty strings, exceeds max tokens, invalid types).

    Example:
        >>> client.embed(model="voyage-3", input="")
        InvalidInputError: Input cannot be empty

    """

    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize InvalidInputError.

        Args:
            message: Error message
            parameter: Name of the invalid parameter
            details: Optional additional error details

        """
        self.parameter = parameter

        details = details or {}
        if parameter:
            details["parameter"] = parameter

        super().__init__(message, details)


class TimeoutError(ProviderError):
    """Exception raised when a request times out.

    Raised when a provider request exceeds the configured timeout.
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        timeout: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize TimeoutError.

        Args:
            message: Error message
            provider: Provider name
            timeout: Timeout value in seconds
            details: Optional additional error details

        """
        self.timeout = timeout

        details = details or {}
        if timeout:
            details["timeout"] = timeout
            message = f"{message} (timeout: {timeout}s)"

        super().__init__(
            message=message,
            provider=provider,
            status_code=408,
            details=details,
        )


class NetworkError(ProviderError):
    """Exception raised for network-related errors.

    Raised when a network error occurs (connection refused, DNS failure, etc.).
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize NetworkError.

        Args:
            message: Error message
            provider: Provider name
            details: Optional additional error details

        """
        super().__init__(
            message=message,
            provider=provider,
            details=details,
        )
