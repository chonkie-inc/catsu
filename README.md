<div align="center">

![Catsu Logo](./assets/catsu-logo-w-bg.png)

# 🌐 catsu 🐱

[![PyPI version](https://img.shields.io/pypi/v/catsu.svg)](https://pypi.org/project/catsu/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://pypi.org/project/catsu/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.catsu.dev)
[![Stars](https://img.shields.io/github/stars/chonkie-inc/catsu?style=social)](https://github.com/chonkie-inc/catsu)

_A unified, batteries-included client for embedding APIs that actually works._

</div>

**The world of embedding API clients is broken.** ([details](./docs/spilled-milk.md))

- [Everyone defaults to OpenAI's client](./docs/spilled-milk.md#openais-client-wasnt-designed-for-embeddings) for embeddings, even though it wasn't designed for that purpose
- [Provider-specific libraries](./docs/spilled-milk.md#provider-specific-libraries-are-inconsistent-or-broken) (VoyageAI, Cohere, etc.) are inconsistent, poorly maintained, or outright broken
- [Universal clients like LiteLLM and any-llm-sdk](./docs/spilled-milk.md#universal-clients-dont-focus-on-embeddings) don't focus on embeddings at all—they rely on native client libraries, inheriting all their problems
- [Every provider has different capabilities](./docs/spilled-milk.md#capability-inconsistencies-across-providers)—some support dimension changes, others don't—with no standardized way to discover what's available
- [Most clients lack basic features](./docs/spilled-milk.md#missing-basic-features) like retry logic, proper error handling, and usage tracking
- [There's no single source of truth](./docs/spilled-milk.md#no-single-source-of-truth-for-model-metadata) for model metadata, pricing, or capabilities

**Catsu fixes this.** It's a lightweight, unified client built specifically for embeddings with:

🎯 A clean, consistent API across all providers </br>
🔄 Built-in retry logic with exponential backoff </br>
💰 Automatic usage and cost tracking </br>
📚 Rich model metadata and capability discovery </br>
⚠️ Proper error handling and type hints </br>
⚡ First-class support for both sync and async

## 📦 Install

Install with uv (recommended):
```bash
uv pip install catsu
```

Or with pip:
```bash
pip install catsu
```

## 🚀 Quick Start

Get started in seconds! Just import catsu, create a client, and start embedding:

```python
import catsu

# Initialize the client
client = catsu.Client()

# Generate embeddings (auto-detects provider from model name)
response = client.embed(
    model="voyage-3",
    input="Hello, embeddings!"
)

# Access your results
print(f"Dimensions: {response.dimensions}")
print(f"Tokens used: {response.usage.tokens}")
print(f"Cost: ${response.usage.cost:.6f}")
print(f"Embedding: {response.embeddings[0][:5]}...")  # First 5 dims
```

That's it! No configuration needed—catsu picks up your API keys from environment variables automatically (`VOYAGE_API_KEY`, `OPENAI_API_KEY`, etc.).

**Want more control?** Specify the provider explicitly:
```python
# Method 1: Separate parameters
response = client.embed(provider="voyageai", model="voyage-3", input="Hello!")

# Method 2: Provider prefix
response = client.embed(model="voyageai:voyage-3", input="Hello!")
```

**Need async?** Just use `aembed`:
```python
response = await client.aembed(model="voyage-3", input="Hello, async world!")
```

📖 **Want to learn more?** Check out the [complete documentation](https://docs.catsu.dev) for detailed guides on all providers, parameters, and best practices.

## 🤝 Contributing

Can't find your favorite model or provider? **Open an issue** and we will promptly try to add it! We're constantly expanding support for new embedding providers and models.

For guidelines on contributing, please see [CONTRIBUTING.md](./CONTRIBUTING.md).

---

<div align="center">

If you found this helpful, consider giving it a ⭐!

made with ❤️ by [chonkie, inc.](https://chonkie.ai)

</div>
