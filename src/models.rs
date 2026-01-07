//! Data models for catsu client.

use serde::{Deserialize, Serialize};

/// Response from an embedding request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedResponse {
    /// The generated embeddings, one per input.
    pub embeddings: Vec<Vec<f32>>,
    /// The model used to generate embeddings.
    pub model: String,
    /// The provider that generated the embeddings.
    pub provider: String,
    /// The dimensionality of the embeddings.
    pub dimensions: usize,
    /// Number of inputs that were embedded.
    pub input_count: usize,
    /// The input type used (if specified).
    pub input_type: Option<InputType>,
    /// Latency of the request in milliseconds.
    pub latency_ms: f64,
    /// Usage information.
    pub usage: Usage,
}

/// Token and cost usage information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Total tokens used across all inputs.
    pub tokens: u64,
    /// Estimated cost in USD (if available).
    pub cost: Option<f64>,
}

/// Request for generating embeddings.
#[derive(Debug, Clone, Serialize)]
pub struct EmbedRequest {
    /// The model to use.
    pub model: String,
    /// Input texts to embed.
    pub inputs: Vec<String>,
    /// Optional input type hint (query or document).
    pub input_type: Option<InputType>,
    /// Optional output dimensions (if model supports it).
    pub dimensions: Option<u32>,
    /// Optional API key override for this request.
    #[serde(skip)]
    pub api_key: Option<String>,
}

/// Input type hint for embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputType {
    /// Query text (for retrieval).
    Query,
    /// Document text (for indexing).
    Document,
}

/// Model information from the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name.
    pub name: String,
    /// Provider name.
    pub provider: String,
    /// Output embedding dimensions.
    pub dimensions: u32,
    /// Maximum input tokens.
    pub max_tokens: u32,
    /// Whether the model supports custom dimensions.
    pub supports_dimensions: bool,
    /// Whether the model supports input_type parameter.
    pub supports_input_type: bool,
    /// Cost per 1M tokens in USD.
    pub cost_per_million_tokens: Option<f64>,
}
