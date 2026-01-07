# Catsu

High-performance embeddings client for multiple providers, powered by Rust.

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
```

## Async Support

```python
import asyncio
from catsu import Client

async def main():
    client = Client()
    response = await client.aembed("openai:text-embedding-3-small", "Hello!")
    print(response.embeddings[0][:5])

asyncio.run(main())
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

## Supported Providers

- OpenAI (`OPENAI_API_KEY`)
- VoyageAI (`VOYAGE_API_KEY`)
- Cohere (`COHERE_API_KEY`)
- Jina (`JINA_API_KEY`)
- Mistral (`MISTRAL_API_KEY`)
- Gemini (`GOOGLE_API_KEY` or `GEMINI_API_KEY`)
- Together (`TOGETHER_API_KEY`)
- Mixedbread (`MIXEDBREAD_API_KEY`)
- Nomic (`NOMIC_API_KEY`)
- DeepInfra (`DEEPINFRA_API_KEY`)
- Cloudflare (`CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID`)

## Features

- **11 providers** with unified API
- **Async support** with `aembed()`
- **Model catalog** with 64+ models
- **Automatic retry** with exponential backoff
- **Cost tracking** per request
- **High performance** - Rust core with Python bindings
