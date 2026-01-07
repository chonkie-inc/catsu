//! Together AI embeddings provider.

use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::errors::ClientError;
use crate::http::HttpClient;
use crate::models::{EmbedRequest, EmbedResponse, Usage};
use crate::providers::EmbeddingProvider;

const TOGETHER_API_URL: &str = "https://api.together.xyz/v1/embeddings";

/// Together AI embeddings provider.
#[derive(Debug)]
pub struct TogetherProvider {
    api_key: String,
    http_client: HttpClient,
}

impl TogetherProvider {
    /// Create a new Together provider with the given API key.
    pub fn new(api_key: String, http_client: HttpClient) -> Self {
        Self {
            api_key,
            http_client,
        }
    }

    /// Create a new Together provider from environment variable.
    pub fn from_env(http_client: HttpClient) -> Result<Self, ClientError> {
        let api_key =
            std::env::var("TOGETHER_API_KEY").map_err(|_| ClientError::MissingApiKey {
                provider: "together".to_string(),
            })?;
        Ok(Self::new(api_key, http_client))
    }
}

/// Together API request body (OpenAI-compatible).
#[derive(Debug, Serialize)]
struct TogetherEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

/// Together API response.
#[derive(Debug, Deserialize)]
struct TogetherEmbeddingResponse {
    data: Vec<TogetherEmbedding>,
    model: String,
    usage: TogetherUsage,
}

#[derive(Debug, Deserialize)]
struct TogetherEmbedding {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct TogetherUsage {
    total_tokens: u64,
}

/// Together API error response.
#[derive(Debug, Deserialize)]
struct TogetherErrorResponse {
    error: TogetherError,
}

#[derive(Debug, Deserialize)]
struct TogetherError {
    message: String,
}

#[async_trait]
impl EmbeddingProvider for TogetherProvider {
    fn name(&self) -> &'static str {
        "together"
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, ClientError> {
        debug!(
            model = %request.model,
            inputs = request.inputs.len(),
            "Sending Together embedding request"
        );

        // Together AI doesn't support input_type or dimensions
        let input_type_value = request.input_type;
        if input_type_value.is_some() {
            debug!("Together AI doesn't use input_type parameter, ignoring");
        }
        if request.dimensions.is_some() {
            debug!("Together AI doesn't support custom dimensions, ignoring");
        }

        let input_count = request.inputs.len();
        let body = TogetherEmbeddingRequest {
            model: &request.model,
            input: &request.inputs,
        };

        let body_json = serde_json::to_string(&body)?;
        // Use request's api_key if provided, otherwise use provider's api_key
        let api_key = request.api_key.as_ref().unwrap_or(&self.api_key).clone();

        let start = Instant::now();
        let response = self
            .http_client
            .send_with_retry(|client| {
                client
                    .post(TOGETHER_API_URL)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("Content-Type", "application/json")
                    .body(body_json.clone())
            })
            .await?;

        let status = response.status().as_u16();
        let response_text = response.text().await?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        if status != 200 {
            if let Ok(error_response) =
                serde_json::from_str::<TogetherErrorResponse>(&response_text)
            {
                return Err(ClientError::Api {
                    status,
                    message: error_response.error.message,
                });
            }
            return Err(ClientError::Api {
                status,
                message: response_text,
            });
        }

        let together_response: TogetherEmbeddingResponse = serde_json::from_str(&response_text)?;

        // Sort embeddings by index
        let mut embeddings: Vec<_> = together_response.data.into_iter().collect();
        embeddings.sort_by_key(|e| e.index);

        let embedding_vectors: Vec<Vec<f32>> =
            embeddings.into_iter().map(|e| e.embedding).collect();

        let dimensions = embedding_vectors.first().map(|e| e.len()).unwrap_or(0);
        let total_tokens = together_response.usage.total_tokens;

        let cost = calculate_cost(&request.model, total_tokens);

        Ok(EmbedResponse {
            embeddings: embedding_vectors,
            model: together_response.model,
            provider: "together".to_string(),
            dimensions,
            input_count,
            input_type: input_type_value,
            latency_ms,
            usage: Usage {
                tokens: total_tokens,
                cost,
            },
        })
    }
}

fn calculate_cost(model: &str, tokens: u64) -> Option<f64> {
    // Together AI embedding pricing
    let price_per_million = match model {
        m if m.contains("bge") => 0.008,
        m if m.contains("UAE") => 0.008,
        m if m.contains("M2-BERT") => 0.008,
        _ => 0.008, // Default Together embedding price
    };

    Some((tokens as f64 / 1_000_000.0) * price_per_million)
}
