"""Catsu - High-performance embeddings client for multiple providers.

This is a Rust-powered client with Python bindings.

Supported providers:
- OpenAI (OPENAI_API_KEY)
- VoyageAI (VOYAGE_API_KEY)
- Cohere (COHERE_API_KEY)
- Jina (JINA_API_KEY)
- Mistral (MISTRAL_API_KEY)
- Gemini (GOOGLE_API_KEY or GEMINI_API_KEY)
- Together (TOGETHER_API_KEY)
- Mixedbread (MIXEDBREAD_API_KEY)
- Nomic (NOMIC_API_KEY)
- DeepInfra (DEEPINFRA_API_KEY)
- Cloudflare (CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID)

Example:
    >>> from catsu import Client
    >>> client = Client()
    >>> response = client.embed("openai:text-embedding-3-small", "Hello, world!")
    >>> print(response.embeddings[0][:5])

    # Async embedding:
    >>> response = await client.aembed("openai:text-embedding-3-small", "Hello")

    # List available models:
    >>> models = client.list_models("openai")
    >>> for m in models:
    ...     print(f"{m.name}: {m.dimensions} dims")

    # Convert to numpy:
    >>> arr = response.to_numpy()
    >>> print(arr.shape)  # (1, 1536)
"""

from catsu._catsu import Client, EmbedResponse, ModelInfo, Usage

__all__ = ["Client", "EmbedResponse", "ModelInfo", "Usage"]
__version__ = "0.1.5"
