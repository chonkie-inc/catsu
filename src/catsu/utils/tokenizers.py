"""Tokenizer utilities for Catsu.

This module provides a unified interface for loading tokenizers from different
backends (HuggingFace tokenizers and tiktoken).
"""

from typing import Any, Dict, List, Protocol, Union


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer objects."""

    def encode(self, text: str) -> Any:
        """Encode text to tokens."""
        ...


class HuggingFaceTokenizerWrapper:
    """Wrapper for HuggingFace tokenizers to provide consistent interface."""

    def __init__(self, tokenizer: Any) -> None:
        """Initialize with a HuggingFace tokenizer."""
        self._tokenizer = tokenizer

    def encode(self, text: str) -> List[int]:
        """Encode text and return token IDs."""
        encoding = self._tokenizer.encode(text)
        return encoding.ids

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encode(text))


class TiktokenWrapper:
    """Wrapper for tiktoken to provide consistent interface."""

    def __init__(self, encoding: Any) -> None:
        """Initialize with a tiktoken encoding."""
        self._encoding = encoding

    def encode(self, text: str) -> List[int]:
        """Encode text and return token IDs."""
        return self._encoding.encode(text)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encode(text))


def load_tokenizer(
    config: Dict[str, Any],
) -> Union[HuggingFaceTokenizerWrapper, TiktokenWrapper]:
    """Load a tokenizer based on configuration.

    The config dict should have:
    - {"engine": "tiktoken", "name": "cl100k_base"} for tiktoken encodings
    - {"engine": "huggingface", "name": "org/model"} for HuggingFace tokenizers

    Args:
        config: Tokenizer configuration dictionary with 'engine' and 'name' keys

    Returns:
        Wrapped tokenizer with consistent interface

    Raises:
        ImportError: If required library is not installed
        ValueError: If config is invalid

    Example:
        >>> # Load HuggingFace tokenizer
        >>> tokenizer = load_tokenizer({"engine": "huggingface", "name": "voyageai/voyage-3"})
        >>> tokens = tokenizer.encode("hello world")

        >>> # Load tiktoken encoding
        >>> tokenizer = load_tokenizer({"engine": "tiktoken", "name": "cl100k_base"})
        >>> tokens = tokenizer.encode("hello world")

    """
    if not config:
        raise ValueError("Tokenizer config cannot be empty")

    engine = config.get("engine")
    name = config.get("name")

    if not engine:
        raise ValueError("Tokenizer config must have 'engine' field")
    if not name:
        raise ValueError("Tokenizer config must have 'name' field")

    # Load tiktoken
    if engine == "tiktoken":
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(
                "tiktoken library is required for this tokenizer. "
                "Install it with: uv add tiktoken"
            ) from e

        try:
            encoding = tiktoken.get_encoding(name)
            return TiktokenWrapper(encoding)
        except Exception as e:
            raise ValueError(f"Failed to load tiktoken encoding '{name}': {e}") from e

    # Load HuggingFace tokenizers
    if engine == "huggingface":
        try:
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "tokenizers library is required for this tokenizer. "
                "Install it with: uv add tokenizers"
            ) from e

        try:
            tokenizer = Tokenizer.from_pretrained(name)
            return HuggingFaceTokenizerWrapper(tokenizer)
        except Exception as e:
            raise ValueError(
                f"Failed to load HuggingFace tokenizer from '{name}': {e}"
            ) from e

    raise ValueError(
        f"Unknown tokenizer engine '{engine}'. Must be 'tiktoken' or 'huggingface'"
    )


def count_tokens(text: str, config: Dict[str, Any]) -> int:
    """Count tokens in text using the specified tokenizer.

    Convenience function that loads tokenizer and counts tokens.

    Args:
        text: Text to tokenize
        config: Tokenizer configuration

    Returns:
        Number of tokens

    Example:
        >>> count = count_tokens("hello world", {"name": "cl100k_base"})
        >>> print(count)
        2

    """
    tokenizer = load_tokenizer(config)
    return tokenizer.count_tokens(text)
