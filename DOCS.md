<div align="center">

# Catsu Documentation

**A unified Python client for embedding APIs**

</div>

---

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Client API](#client-api)
  - [Initialization](#initialization)
  - [Methods](#methods)
  - [Context Manager Usage](#context-manager-usage)
- [Provider Documentation](#provider-documentation)
  - [Cohere](#cohere)
  - [Gemini](#gemini)
  - [Jina AI](#jina-ai)
  - [Mistral AI](#mistral-ai)
  - [Nomic](#nomic)
  - [OpenAI](#openai)
  - [Voyage AI](#voyage-ai)
- [Models Reference](#models-reference)
  - [Cohere Models](#cohere-models)
  - [Gemini Models](#gemini-models)
  - [Jina AI Models](#jina-ai-models)
  - [Mistral AI Models](#mistral-ai-models)
  - [Nomic Models](#nomic-models)
  - [OpenAI Models](#openai-models)
  - [Voyage AI Models](#voyage-ai-models)
- [Common Parameters](#common-parameters)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

---

## Introduction

Catsu is a unified Python client for accessing multiple embedding providers through a single, consistent API. It supports 7 major embedding providers with over 35 models, providing features like:

- **Unified Interface**: Single API for all providers
- **Automatic Retry Logic**: Built-in exponential backoff
- **Cost Tracking**: Automatic usage and cost calculation
- **Local Tokenization**: Count tokens without API calls
- **Type Safety**: Full type hints and validation
- **Async Support**: Both sync and async methods

---

## Installation

```bash
pip install catsu
```

---

## Quick Start

```python
import catsu

# Initialize client
client = catsu.Client()

# Generate embeddings
response = client.embed(
    model="voyage-3",
    input="Hello, world!",
    provider="voyageai"  # Optional if model name is unique
)

# Access results
print(response.embeddings)      # [[0.1, 0.2, ...]]
print(response.usage.tokens)    # 2
print(response.usage.cost)      # 0.0000002
```

---

## Client API

### Initialization

```python
from catsu import Client

client = Client(
    verbose=False,           # Enable verbose logging
    max_retries=3,           # Maximum retry attempts
    timeout=30,              # Request timeout in seconds
    api_keys={               # Optional: API keys by provider
        "voyageai": "key123",
        "openai": "key456"
    }
)
```

**Environment Variables:**
- `VOYAGE_API_KEY` - Voyage AI API key
- `OPENAI_API_KEY` - OpenAI API key
- `COHERE_API_KEY` - Cohere API key
- `GEMINI_API_KEY` - Gemini API key
- `JINA_API_KEY` - Jina AI API key
- `MISTRAL_API_KEY` - Mistral AI API key
- `NOMIC_API_KEY` - Nomic API key

### Methods

#### `embed()`

Generate embeddings synchronously.

```python
response = client.embed(
    model: str,                           # Model name or "provider:model"
    input: Union[str, List[str]],        # Text(s) to embed
    provider: Optional[str] = None,       # Provider name (if not in model)
    input_type: Optional[str] = None,     # "query" or "document"
    dimensions: Optional[int] = None,     # Output dimensions (if supported)
    api_key: Optional[str] = None,        # Per-request API key override
    **kwargs                              # Provider-specific parameters
)
```

**Three ways to specify the model:**

```python
# 1. Explicit provider parameter
client.embed(provider="voyageai", model="voyage-3", input="hello")

# 2. Provider prefix in model name
client.embed(model="voyageai:voyage-3", input="hello")

# 3. Auto-detection (if model name is unique)
client.embed(model="voyage-3", input="hello")
```

#### `aembed()`

Generate embeddings asynchronously.

```python
import asyncio

async def main():
    response = await client.aembed(
        model="voyage-3",
        input="Hello, world!",
        provider="voyageai"
    )
    print(response.embeddings)

asyncio.run(main())
```

#### `list_models()`

List available models, optionally filtered by provider.

```python
# List all models
all_models = client.list_models()

# List models for a specific provider
voyageai_models = client.list_models(provider="voyageai")

for model in voyageai_models:
    print(f"{model['name']}: {model['dimensions']} dims, ${model['cost_per_million_tokens']}/M tokens")
```

### Context Manager Usage

```python
# Automatic cleanup
with catsu.Client() as client:
    response = client.embed(model="voyage-3", input="hello")
    print(response.embeddings)

# Async context manager
async with catsu.Client() as client:
    response = await client.aembed(model="voyage-3", input="hello")
    print(response.embeddings)
```

---

## Provider Documentation

### Cohere

**API Documentation**: https://docs.cohere.com/reference/embed

**Supported Parameters:**

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `model` | str | Model name | ✓ |
| `inputs` | List[str] | Input texts | ✓ |
| `input_type` | Optional[str] | "query" or "document" (mapped to Cohere's API) | ✗ |
| `truncate` | Optional[str] | "NONE", "START", or "END" | ✗ |
| `api_key` | Optional[str] | Per-request API key override | ✗ |

**Note**: Currently only supports `float` embeddings. Binary and int8 quantization not yet supported.

**Example:**

```python
response = client.embed(
    provider="cohere",
    model="embed-v4.0",
    input=["search query", "document text"],
    input_type="query",
    truncate="END"
)
```

**Input Type Mapping:**
- `"query"` → `"search_query"`
- `"document"` → `"search_document"`

---

### Gemini

**API Documentation**: https://ai.google.dev/gemini-api/docs/embeddings

**Supported Parameters:**

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `model` | str | Model name | ✓ |
| `inputs` | List[str] | Input texts | ✓ |
| `input_type` | Optional[str] | "query" or "document" | ✗ |
| `task_type` | Optional[str] | Gemini-specific task type | ✗ |
| `dimensions` | Optional[int] | Output dimensions (128-3072) | ✗ |
| `api_key` | Optional[str] | Per-request API key override | ✗ |

**Valid Task Types:**
- `"RETRIEVAL_QUERY"`
- `"RETRIEVAL_DOCUMENT"`
- `"SEMANTIC_SIMILARITY"`
- `"CLASSIFICATION"`
- `"CLUSTERING"`
- `"QUESTION_ANSWERING"`
- `"FACT_VERIFICATION"`
- `"CODE_RETRIEVAL_QUERY"`

**Example:**

```python
response = client.embed(
    provider="gemini",
    model="gemini-embedding-001",
    input=["What is machine learning?"],
    task_type="RETRIEVAL_QUERY",
    dimensions=768
)
```

**Features:**
- Matryoshka embeddings (flexible dimensionality)
- Long context support (2048 tokens)

---

### Jina AI

**API Documentation**: https://jina.ai/embeddings/

**Supported Parameters:**

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `model` | str | Model name | ✓ |
| `inputs` | List[str] | Input texts | ✓ |
| `input_type` | Optional[str] | "query" or "document" | ✗ |
| `task` | Optional[str] | Task type for embeddings | ✗ |
| `dimensions` | Optional[int] | Output dimensions (Matryoshka) | ✗ |
| `normalized` | bool | L2 normalize embeddings (default: True) | ✗ |
| `api_key` | Optional[str] | Per-request API key override | ✗ |

**Valid Task Types:**
- `"retrieval.query"`
- `"retrieval.passage"`
- `"text-matching"`
- `"classification"`
- `"separation"`

**Example:**

```python
response = client.embed(
    provider="jinaai",
    model="jina-embeddings-v4",
    input=["search query"],
    task="retrieval.query",
    dimensions=1024,
    normalized=True
)
```

**Features:**
- Multimodal support (text + image for v4)
- Long context (up to 32,768 tokens)
- Matryoshka embeddings
- Code-specific models

---

### Mistral AI

**API Documentation**: https://docs.mistral.ai/capabilities/embeddings

**Supported Parameters:**

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `model` | str | Model name | ✓ |
| `inputs` | List[str] | Input texts (up to 512) | ✓ |
| `input_type` | Optional[str] | "query" or "document" | ✗ |
| `encoding_format` | Optional[str] | Output format (e.g., "float", "int8") | ✗ |
| `dimensions` | Optional[int] | Output dimensions (codestral-embed-2505) | ✗ |
| `api_key` | Optional[str] | Per-request API key override | ✗ |

**Example:**

```python
response = client.embed(
    provider="mistral",
    model="codestral-embed-2505",
    input=["def hello(): print('world')"],
    encoding_format="float",
    dimensions=1024
)
```

**Features:**
- Batch embeddings (up to 512 texts)
- Code-optimized models
- Binary quantization support (codestral-embed-2505)

---

### Nomic

**API Documentation**: https://docs.nomic.ai/reference/api/embed-text-v-1-embedding-text-post

**Supported Parameters:**

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `model` | str | Model name | ✓ |
| `inputs` | List[str] | Input texts | ✓ |
| `input_type` | Optional[str] | "query" or "document" | ✗ |
| `task_type` | Optional[str] | Nomic-specific task type | ✗ |
| `dimensions` | Optional[int] | Output dimensions (64-768, v1.5 only) | ✗ |
| `long_text_mode` | str | "truncate" or "mean" (default: "mean") | ✗ |
| `max_tokens_per_text` | int | Max tokens per text (default: 8192) | ✗ |
| `api_key` | Optional[str] | Per-request API key override | ✗ |

**Valid Task Types:**
- `"search_document"`
- `"search_query"`
- `"clustering"`
- `"classification"`

**Example:**

```python
response = client.embed(
    provider="nomic",
    model="nomic-embed-text-v1.5",
    input=["long document text here..."],
    task_type="search_document",
    dimensions=256,
    long_text_mode="mean",
    max_tokens_per_text=8192
)
```

**Input Type Mapping:**
- `"query"` → `"search_query"`
- `"document"` → `"search_document"`

**Features:**
- Long context support (8192 tokens)
- Matryoshka embeddings (v1.5)
- Mean pooling for long texts

---

### OpenAI

**API Documentation**: https://platform.openai.com/docs/api-reference/embeddings

**Supported Parameters:**

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `model` | str | Model name | ✓ |
| `inputs` | List[str] | Input texts | ✓ |
| `input_type` | Optional[str] | **Ignored** (OpenAI doesn't support this) | ✗ |
| `dimensions` | Optional[int] | Output dimensions (text-embedding-3 only) | ✗ |
| `api_key` | Optional[str] | Per-request API key override | ✗ |

**Example:**

```python
response = client.embed(
    provider="openai",
    model="text-embedding-3-small",
    input=["hello world"],
    dimensions=512  # Only for text-embedding-3 models
)
```

**Features:**
- Industry-standard performance
- Flexible dimensions (text-embedding-3 models)
- Cost-effective pricing

**Note**: OpenAI does not support `input_type`. The parameter is accepted but ignored.

---

### Voyage AI

**API Documentation**: https://docs.voyageai.com/docs/embeddings

**Supported Parameters:**

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `model` | str | Model name | ✓ |
| `inputs` | List[str] | Input texts | ✓ |
| `input_type` | Optional[str] | "query" or "document" | ✗ |
| `dimensions` | Optional[int] | Output dimensions (256, 512, 1024, 2048) | ✗ |
| `api_key` | Optional[str] | Per-request API key override | ✗ |

**Example:**

```python
response = client.embed(
    provider="voyageai",
    model="voyage-3",
    input=["search query"],
    input_type="query",
    dimensions=512
)
```

**Features:**
- State-of-the-art performance
- Domain-specific models (code, finance, law)
- Multimodal support (voyage-multimodal-3)
- Binary quantization (voyage-3.5)

---

## Models Reference

### Cohere Models

| Model | Dimensions | Max Tokens | Cost/M | MTEB Score | Features |
|-------|------------|------------|--------|------------|----------|
| `embed-v4.0` | 1536 | 128000 | $0.12 | 64.98 | Latest, multilingual, quantization support |
| `embed-english-v3.0` | 1024 | 512 | $0.10 | 66.21 | English-optimized |
| `embed-english-light-v3.0` | 384 | 512 | $0.10 | 64.59 | Lightweight English |
| `embed-multilingual-v3.0` | 1024 | 512 | $0.10 | 66.01 | Multilingual support |
| `embed-multilingual-light-v3.0` | 384 | 512 | $0.10 | 62.75 | Lightweight multilingual |

**Quantization Support** (embed-v4.0):
- Float (default)
- Int8
- Binary

---

### Gemini Models

| Model | Dimensions | Max Tokens | Cost/M | MTEB Score | Features |
|-------|------------|------------|--------|------------|----------|
| `gemini-embedding-001` | 3072 | 2048 | $0.15 | 73.3 | Matryoshka, task-specific |

**Supported Dimensions**: 128-3072 (flexible)

---

### Jina AI Models

| Model | Dimensions | Max Tokens | Cost/M | Features |
|-------|------------|------------|--------|----------|
| `jina-embeddings-v4` | 2048 | 32768 | $0.05 | Multimodal (text+image), Matryoshka |
| `jina-embeddings-v3` | 1024 | 8194 | $0.05 | Text-only, task-specific |
| `jina-code-embeddings-1.5b` | 1536 | 32768 | $0.05 | Code-optimized, Matryoshka |
| `jina-code-embeddings-0.5b` | 896 | 32768 | $0.05 | Lightweight code model |
| `jina-embeddings-v2-base-en` | 768 | 8192 | $0.05 | English-only |
| `jina-embeddings-v2-base-code` | 768 | 8192 | $0.05 | Code-specific |

---

### Mistral AI Models

| Model | Dimensions | Max Tokens | Cost/M | Features |
|-------|------------|------------|--------|----------|
| `codestral-embed-2505` | 1536 | 32768 | $0.15 | Code-optimized, quantization, Matryoshka |
| `mistral-embed` | 1024 | 8192 | $0.01 | General-purpose, cost-effective |

**Quantization Support** (codestral-embed-2505):
- Float (default)
- Int8
- Binary

---

### Nomic Models

| Model | Dimensions | Max Tokens | Cost/M | MTEB Score | Features |
|-------|------------|------------|--------|------------|----------|
| `nomic-embed-text-v1.5` | 768 | 8192 | $0.10 | 62.2 | Matryoshka (64-768), long context |
| `nomic-embed-text-v1` | 768 | 8192 | $0.10 | 63.6 | Standard embedding |

---

### OpenAI Models

| Model | Dimensions | Max Tokens | Cost/M | MTEB Score | Features |
|-------|------------|------------|--------|------------|----------|
| `text-embedding-3-large` | 3072 | 8191 | $0.13 | 79.15 | Best performance, flexible dimensions |
| `text-embedding-3-small` | 1536 | 8191 | $0.02 | 77.56 | Cost-effective, flexible dimensions |
| `text-embedding-ada-002` | 1536 | 8191 | $0.10 | 75.79 | Legacy model |

**Flexible Dimensions**: text-embedding-3 models support custom dimensions

---

### Voyage AI Models

| Model | Dimensions | Max Tokens | Cost/M | Features |
|-------|------------|------------|--------|----------|
| `voyage-3-large` | 1024 | 32000 | $0.18 | Premium performance |
| `voyage-3.5` | 1024 | 32000 | $0.06 | Latest general-purpose, quantization |
| `voyage-3.5-lite` | 1024 | 32000 | $0.02 | Cost-effective |
| `voyage-3` | 1024 | 32000 | $0.06 | General-purpose |
| `voyage-3-lite` | 1024 | 32000 | $0.02 | Lightweight |
| `voyage-code-3` | 1024 | 32000 | $0.18 | Code-optimized |
| `voyage-finance-2` | 1024 | 32000 | $0.12 | Finance-specific |
| `voyage-law-2` | 1024 | 16000 | $0.12 | Legal-specific |
| `voyage-code-2` | 1536 | 16000 | $0.12 | Legacy code model |
| `voyage-multilingual-2` | 1024 | 32000 | $0.12 | Multilingual |
| `voyage-multimodal-3` | 1024 | 32000 | $0.12 | Multimodal (text+image) |

**Quantization Support** (voyage-3.5):
- Float (default)
- Int8
- Binary

---

## Common Parameters

### input_type

Specifies the intended use of the input text. Helps models optimize embeddings for specific tasks.

**Values:**
- `"query"`: Short search queries or questions
- `"document"`: Longer documents or passages to be searched

**Example:**

```python
# Embedding a search query
query_embedding = client.embed(
    model="voyage-3",
    input="what is machine learning",
    input_type="query"
)

# Embedding documents
doc_embeddings = client.embed(
    model="voyage-3",
    input=["Machine learning is...", "Deep learning involves..."],
    input_type="document"
)
```

**Provider Support:**
- ✓ Cohere (mapped to `search_query`/`search_document`)
- ✓ Gemini
- ✓ Jina AI
- ✓ Mistral AI
- ✓ Nomic (mapped to `search_query`/`search_document`)
- ✗ OpenAI (parameter ignored)
- ✓ Voyage AI

---

### dimensions

Specifies custom output dimensions for models that support Matryoshka embeddings.

**Example:**

```python
response = client.embed(
    model="text-embedding-3-small",
    input="hello world",
    dimensions=512  # Default is 1536
)
```

**Provider Support:**
- ✗ Cohere
- ✓ Gemini (128-3072)
- ✓ Jina AI (model-dependent)
- ✓ Mistral AI (codestral-embed-2505)
- ✓ Nomic (64-768, v1.5 only)
- ✓ OpenAI (text-embedding-3 models only)
- ✓ Voyage AI (256, 512, 1024, 2048)

---

### api_key

Override the API key for a specific request.

**Example:**

```python
response = client.embed(
    model="voyage-3",
    input="hello",
    api_key="different-api-key-for-this-request"
)
```

All providers support per-request API key overrides.

---

## Error Handling

Catsu provides structured error handling with specific exception types:

```python
from catsu import Client
from catsu.utils.errors import (
    CatsuError,              # Base exception
    AuthenticationError,      # Invalid API key
    RateLimitError,          # Rate limit exceeded
    InvalidInputError,       # Invalid parameters
    ModelNotFoundError,      # Model not found
    UnsupportedFeatureError, # Feature not supported
    ProviderError,           # Provider-specific error
    NetworkError,            # Network issues
    TimeoutError             # Request timeout
)

client = Client()

try:
    response = client.embed(
        model="voyage-3",
        input="hello world"
    )
except AuthenticationError as e:
    print(f"Invalid API key: {e}")
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after} seconds")
except InvalidInputError as e:
    print(f"Invalid input: {e}")
except ModelNotFoundError as e:
    print(f"Model not found: {e}")
except UnsupportedFeatureError as e:
    print(f"Feature not supported: {e}")
except NetworkError as e:
    print(f"Network error: {e}")
except TimeoutError as e:
    print(f"Request timed out: {e}")
except ProviderError as e:
    print(f"Provider error: {e}")
except CatsuError as e:
    print(f"Catsu error: {e}")
```

---

## Best Practices

### 1. Batch Processing

Process multiple texts in a single request for better efficiency:

```python
# Good: Single request for multiple texts
texts = ["text 1", "text 2", "text 3"]
response = client.embed(model="voyage-3", input=texts)

# Avoid: Multiple requests for individual texts
for text in texts:
    response = client.embed(model="voyage-3", input=text)  # Inefficient
```

### 2. Use Context Managers

Ensure proper cleanup of HTTP connections:

```python
with catsu.Client() as client:
    response = client.embed(model="voyage-3", input="hello")
    # Connections automatically closed
```

### 3. Choose the Right Model

Consider your use case:

- **General retrieval**: `voyage-3`, `text-embedding-3-small`
- **Code search**: `voyage-code-3`, `codestral-embed-2505`, `jina-code-embeddings-1.5b`
- **Multilingual**: `embed-multilingual-v3.0`, `voyage-multilingual-2`
- **Cost-sensitive**: `mistral-embed`, `text-embedding-3-small`, `voyage-3.5-lite`
- **Best performance**: `text-embedding-3-large`, `voyage-3-large`, `gemini-embedding-001`

### 4. Specify input_type

When supported, always specify `input_type` for better quality:

```python
# For search queries
query_emb = client.embed(
    model="voyage-3",
    input="what is AI?",
    input_type="query"
)

# For documents
doc_emb = client.embed(
    model="voyage-3",
    input=["AI is artificial intelligence..."],
    input_type="document"
)
```

### 5. Use Async for High Throughput

For concurrent requests, use async methods:

```python
import asyncio

async def process_batch(texts):
    async with catsu.Client() as client:
        tasks = [
            client.aembed(model="voyage-3", input=text)
            for text in texts
        ]
        return await asyncio.gather(*tasks)

results = asyncio.run(process_batch(texts))
```

### 6. Monitor Usage and Costs

Track token usage and costs:

```python
response = client.embed(model="voyage-3", input="hello")
print(f"Tokens: {response.usage.tokens}")
print(f"Cost: ${response.usage.cost:.8f}")
print(f"Latency: {response.latency_ms:.2f}ms")
```

### 7. Handle Rate Limits

Implement exponential backoff for rate limits:

```python
from catsu.utils.errors import RateLimitError
import time

def embed_with_retry(client, model, input, max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.embed(model=model, input=input)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = e.retry_after or (2 ** attempt)
            print(f"Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time)
```

### 8. Use Dimensions for Efficiency

For supported models, reduce dimensions to save storage and improve speed:

```python
# Full dimensions (higher quality)
full_emb = client.embed(
    model="text-embedding-3-small",
    input="hello"
)  # 1536 dimensions

# Reduced dimensions (faster, less storage)
reduced_emb = client.embed(
    model="text-embedding-3-small",
    input="hello",
    dimensions=512
)  # 512 dimensions
```

### 9. Local Tokenization

Count tokens before making API requests to avoid surprises:

```python
# Get the provider instance
provider = client._get_provider("voyageai")

# Count tokens locally (no API call)
token_response = provider.tokenize(
    model="voyage-3",
    inputs=["This is a long text..."]
)

print(f"Token count: {token_response.token_count}")

# Decide whether to proceed based on token count
if token_response.token_count < 1000:
    response = client.embed(model="voyage-3", input="This is a long text...")
```

### 10. Model Auto-detection

Use auto-detection for cleaner code when model names are unique:

```python
# No need to specify provider
response = client.embed(
    model="voyage-3",  # Automatically detected as voyageai
    input="hello"
)

# List all available models to check uniqueness
models = client.list_models()
```

---

**Happy embedding! 🚀**
