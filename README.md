<div align="center">

![Catsu Logo](./assets/catsu-logo-w-bg.png)

# 🌐 catsu 🐱

[![Crates.io](https://img.shields.io/crates/v/catsu.svg?style=flat&labelColor=black&color=orange)](https://crates.io/crates/catsu)
[![PyPI version](https://img.shields.io/pypi/v/catsu.svg?style=flat&labelColor=black&color=orange)](https://pypi.org/project/catsu/)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg?style=flat&labelColor=black)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-latest-orange.svg?style=flat&labelColor=black)](https://docs.catsu.dev)
[![Discord](https://img.shields.io/badge/discord-join-orange.svg?style=flat&labelColor=black&logo=discord&logoColor=white)](https://discord.gg/vH3SkRqmUz)

_A unified, batteries-included client for embedding APIs that actually works._

[Rust](#-rust) • [Python](./packages/python/README_PYPI.md)

</div>

**The world of embedding API clients is broken.** ([details](./docs/spilled-milk.md))

- [Everyone defaults to OpenAI's client](./docs/spilled-milk.md#openais-client-wasnt-designed-for-embeddings) for embeddings, even though it wasn't designed for that purpose
- [Provider-specific libraries](./docs/spilled-milk.md#provider-specific-libraries-are-inconsistent-or-broken) (VoyageAI, Cohere, etc.) are inconsistent, poorly maintained, or outright broken
- [Universal clients like LiteLLM and any-llm-sdk](./docs/spilled-milk.md#universal-clients-dont-focus-on-embeddings) don't focus on embeddings at all—they rely on native client libraries, inheriting all their problems
- [Every provider has different capabilities](./docs/spilled-milk.md#capability-inconsistencies-across-providers)—some support dimension changes, others don't—with no standardized way to discover what's available
- [Most clients lack basic features](./docs/spilled-milk.md#missing-basic-features) like retry logic, proper error handling, and usage tracking
- [There's no single source of truth](./docs/spilled-milk.md#no-single-source-of-truth-for-model-metadata) for model metadata, pricing, or capabilities

**Catsu fixes this.** It's a high-performance, unified client built specifically for embeddings with:

🎯 A clean, consistent API across all providers </br>
🔄 Built-in retry logic with exponential backoff </br>
💰 Automatic usage and cost tracking </br>
📚 Rich model metadata and capability discovery </br>
⚡ Rust core with Python bindings for maximum performance

## 📦 Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
catsu = "0.1"
tokio = { version = "1", features = ["full"] }
```

### Quick Start

```rust
use catsu::Client;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create client (reads API keys from environment)
    let client = Client::new()?;

    // Generate embeddings
    let response = client.embed(
        "openai:text-embedding-3-small",
        vec!["Hello, world!".to_string(), "How are you?".to_string()],
    ).await?;

    println!("Dimensions: {}", response.dimensions);
    println!("Tokens used: {}", response.usage.tokens);
    println!("Embedding: {:?}", &response.embeddings[0][..5]);

    Ok(())
}
```

### With Options

```rust
use catsu::{Client, InputType};

let response = client.embed_with_options(
    "openai:text-embedding-3-small",
    vec!["Search query".to_string()],
    Some(InputType::Query),  // input type hint
    Some(256),               // output dimensions
).await?;
```

### Model Catalog

```rust
// List all available models
let models = client.list_models(None);

// Filter by provider
let openai_models = client.list_models(Some("openai"));
for model in openai_models {
    println!("{}: {} dims", model.name, model.dimensions);
}
```

## 🐍 Python

Looking for Python? See the **[Python documentation](./packages/python/README_PYPI.md)**.

```bash
pip install catsu
```

```python
from catsu import Client

client = Client()
response = client.embed("openai:text-embedding-3-small", ["Hello, world!"])
print(f"Dimensions: {response.dimensions}")
```

## 🤝 Contributing

Can't find your favorite model or provider? **Open an issue** and we'll add it!

For guidelines on contributing, see [CONTRIBUTING.md](./CONTRIBUTING.md).

---

<div align="center">

If you found this helpful, consider giving it a ⭐!

made with ❤️ by [chonkie, inc.](https://chonkie.ai)

</div>
