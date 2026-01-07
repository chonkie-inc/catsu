//! Catsu - High-performance embeddings client for multiple providers.
//!
//! Catsu provides a unified interface for generating embeddings from various
//! providers like OpenAI, VoyageAI, Cohere, and more.
//!
//! # Quick Start
//!
//! ```no_run
//! use catsu::Client;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create client (reads API keys from environment)
//!     let client = Client::new()?;
//!
//!     // Generate embeddings
//!     let response = client.embed(
//!         "openai:text-embedding-3-small",
//!         vec!["Hello, world!".to_string()],
//!     ).await?;
//!
//!     println!("Dimensions: {}", response.embeddings[0].len());
//!     println!("Tokens used: {}", response.usage.tokens);
//!     Ok(())
//! }
//! ```
//!
//! # Features
//!
//! - **Multiple providers**: OpenAI, VoyageAI, Cohere, Jina, Mistral, and more
//! - **Automatic retry**: Exponential backoff with jitter
//! - **Cost tracking**: Estimates cost per request
//! - **Type-safe**: Full Rust type safety
//!
//! # Supported Providers
//!
//! | Provider | Environment Variable |
//! |----------|---------------------|
//! | OpenAI | `OPENAI_API_KEY` |
//! | VoyageAI | `VOYAGE_API_KEY` |
//! | Cohere | `COHERE_API_KEY` |
//! | Jina | `JINA_API_KEY` |
//! | Mistral | `MISTRAL_API_KEY` |
//! | Gemini | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
//! | Together | `TOGETHER_API_KEY` |
//! | Mixedbread | `MIXEDBREAD_API_KEY` |
//! | Nomic | `NOMIC_API_KEY` |
//! | DeepInfra | `DEEPINFRA_API_KEY` |
//! | Cloudflare | `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` |

pub mod catalog;
pub mod client;
pub mod errors;
pub mod http;
pub mod models;
pub mod providers;

// Re-exports
pub use catalog::{get_model, list_catalog_providers, list_models};
pub use client::Client;
pub use errors::ClientError;
pub use http::{HttpClient, HttpConfig};
pub use models::{EmbedRequest, EmbedResponse, InputType, ModelInfo, Usage};
pub use providers::EmbeddingProvider;
