<div align="center">

![Catsu Logo](https://raw.githubusercontent.com/chonkie-inc/catsu/main/assets/catsu-logo-w-bg.png)

# 🐱 catsu

[![PyPI version](https://img.shields.io/pypi/v/catsu.svg?style=flat&labelColor=black&color=orange)](https://pypi.org/project/catsu/)
[![Python](https://img.shields.io/badge/python-3.10+-orange.svg?style=flat&labelColor=black)](https://pypi.org/project/catsu/)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg?style=flat&labelColor=black)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-latest-orange.svg?style=flat&labelColor=black)](https://docs.catsu.dev)
[![Discord](https://img.shields.io/badge/discord-join-orange.svg?style=flat&labelColor=black&logo=discord&logoColor=white)](https://discord.gg/vH3SkRqmUz)

_A unified, batteries-included client for embedding APIs that actually works._

</div>

**The world of embedding API clients is broken.**

- Everyone defaults to OpenAI's client for embeddings, even though it wasn't designed for that purpose
- Provider-specific libraries (VoyageAI, Cohere, etc.) are inconsistent, poorly maintained, or outright broken
- Universal clients like LiteLLM don't focus on embeddings—they rely on native client libraries, inheriting all their problems
- Every provider has different capabilities—some support dimension changes, others don't—with no standardized way to discover what's available
- Most clients lack basic features like retry logic, proper error handling, and usage tracking

**Catsu fixes this.** It's a high-performance, unified client built specifically for embeddings with:

🎯 A clean, consistent API across all providers </br>
🔄 Built-in retry logic with exponential backoff </br>
💰 Automatic usage and cost tracking </br>
📚 Rich model metadata and capability discovery </br>
⚡ Rust core with Python bindings for maximum performance

## Installation

```bash
pip install catsu
```

## Quick Start

```python
from catsu import Client

# Create client (reads API keys from environment)
client = Client()

# Generate embeddings
response = client.embed(
    "openai:text-embedding-3-small",
    ["Hello, world!", "How are you?"]
)

print(f"Dimensions: {response.dimensions}")
print(f"Tokens used: {response.usage.tokens}")
print(f"Embedding: {response.embeddings[0][:5]}")
```

## Async Support

```python
import asyncio
from catsu import Client

async def main():
    client = Client()
    response = await client.aembed(
        "openai:text-embedding-3-small",
        "Hello, async world!"
    )
    print(response.embeddings[0][:5])

asyncio.run(main())
```

## With Options

```python
response = client.embed(
    "openai:text-embedding-3-small",
    ["Search query"],
    input_type="query",  # "query" or "document"
    dimensions=256,      # output dimensions (if supported)
)
```

## Model Catalog

```python
# List all available models
models = client.list_models()

# Filter by provider
openai_models = client.list_models("openai")
for m in openai_models:
    print(f"{m.name}: {m.dimensions} dims, ${m.cost_per_million_tokens}/M tokens")
```

## Configuration

```python
client = Client(
    max_retries=5,   # Default: 3
    timeout=60,      # Default: 30 seconds
)
```

## NumPy Integration

```python
# Convert embeddings to numpy array
arr = response.to_numpy()
print(arr.shape)  # (2, 1536)
```

## Context Manager

```python
# Sync
with Client() as client:
    response = client.embed("openai:text-embedding-3-small", "Hello!")

# Async
async with Client() as client:
    response = await client.aembed("openai:text-embedding-3-small", "Hello!")
```

---

<div align="center">

If you found this helpful, consider giving it a ⭐!

made with ❤️ by [chonkie, inc.](https://chonkie.ai)

</div>
