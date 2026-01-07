//! Embedding providers.

pub mod cloudflare;
pub mod cohere;
pub mod deepinfra;
pub mod gemini;
pub mod jina;
pub mod mistral;
pub mod mixedbread;
pub mod nomic;
pub mod openai;
pub mod together;
pub mod voyageai;

use async_trait::async_trait;

use crate::errors::ClientError;
use crate::models::{EmbedRequest, EmbedResponse};

pub use cloudflare::CloudflareProvider;
pub use cohere::CohereProvider;
pub use deepinfra::DeepInfraProvider;
pub use gemini::GeminiProvider;
pub use jina::JinaProvider;
pub use mistral::MistralProvider;
pub use mixedbread::MixedbreadProvider;
pub use nomic::NomicProvider;
pub use openai::OpenAIProvider;
pub use together::TogetherProvider;
pub use voyageai::VoyageAIProvider;

/// Trait for embedding providers.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Get the provider name.
    fn name(&self) -> &'static str;

    /// Generate embeddings for the given request.
    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, ClientError>;
}
