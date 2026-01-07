//! OpenAI embeddings provider.

use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::errors::ClientError;
use crate::http::HttpClient;
use crate::models::{EmbedRequest, EmbedResponse, Usage};
use crate::providers::EmbeddingProvider;

const OPENAI_API_URL: &str = "https://api.openai.com/v1/embeddings";

/// OpenAI embeddings provider.
#[derive(Debug)]
pub struct OpenAIProvider {
    api_key: String,
    http_client: HttpClient,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider with the given API key.
    pub fn new(api_key: String, http_client: HttpClient) -> Self {
        Self {
            api_key,
            http_client,
        }
    }

    /// Create a new OpenAI provider from environment variable.
    pub fn from_env(http_client: HttpClient) -> Result<Self, ClientError> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| ClientError::MissingApiKey {
            provider: "openai".to_string(),
        })?;
        Ok(Self::new(api_key, http_client))
    }
}

/// OpenAI API request body.
#[derive(Debug, Serialize)]
struct OpenAIEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

/// OpenAI embeddings API response.
#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbedding>,
    model: String,
    usage: OpenAIUsage,
}

/// Single embedding from OpenAI.
#[derive(Debug, Deserialize)]
struct OpenAIEmbedding {
    embedding: Vec<f32>,
    index: usize,
}

/// OpenAI usage information.
#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    #[allow(dead_code)]
    prompt_tokens: u64,
    total_tokens: u64,
}

/// OpenAI API error response.
#[derive(Debug, Deserialize)]
struct OpenAIErrorResponse {
    error: OpenAIError,
}

/// OpenAI error details.
#[derive(Debug, Deserialize)]
struct OpenAIError {
    message: String,
}

#[async_trait]
impl EmbeddingProvider for OpenAIProvider {
    fn name(&self) -> &'static str {
        "openai"
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, ClientError> {
        debug!(
            model = %request.model,
            inputs = request.inputs.len(),
            "Sending OpenAI embedding request"
        );

        // OpenAI doesn't use input_type, but we accept it for API compatibility
        let input_type = request.input_type;
        if input_type.is_some() {
            debug!("OpenAI doesn't use input_type parameter, ignoring");
        }

        let input_count = request.inputs.len();
        let body = OpenAIEmbeddingRequest {
            model: &request.model,
            input: &request.inputs,
            dimensions: request.dimensions,
        };

        let body_json = serde_json::to_string(&body)?;
        // Use request's api_key if provided, otherwise use provider's api_key
        let api_key = request.api_key.as_ref().unwrap_or(&self.api_key).clone();

        let start = Instant::now();
        let response = self
            .http_client
            .send_with_retry(|client| {
                client
                    .post(OPENAI_API_URL)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("Content-Type", "application/json")
                    .body(body_json.clone())
            })
            .await?;

        let status = response.status().as_u16();
        let response_text = response.text().await?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        if status != 200 {
            // Try to parse error response
            if let Ok(error_response) = serde_json::from_str::<OpenAIErrorResponse>(&response_text)
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

        let openai_response: OpenAIEmbeddingResponse = serde_json::from_str(&response_text)?;

        // Sort embeddings by index to ensure correct order
        let mut embeddings: Vec<_> = openai_response.data.into_iter().collect();
        embeddings.sort_by_key(|e| e.index);

        let embedding_vectors: Vec<Vec<f32>> =
            embeddings.into_iter().map(|e| e.embedding).collect();

        // Get dimensions from first embedding
        let dimensions = embedding_vectors.first().map(|e| e.len()).unwrap_or(0);

        // Calculate cost (OpenAI text-embedding-3-small: $0.02/1M tokens)
        let cost = calculate_cost(&request.model, openai_response.usage.total_tokens);

        Ok(EmbedResponse {
            embeddings: embedding_vectors,
            model: openai_response.model,
            provider: "openai".to_string(),
            dimensions,
            input_count,
            input_type,
            latency_ms,
            usage: Usage {
                tokens: openai_response.usage.total_tokens,
                cost,
            },
        })
    }
}

/// Calculate cost based on model and token count.
fn calculate_cost(model: &str, tokens: u64) -> Option<f64> {
    // Pricing per 1M tokens (as of 2024)
    let price_per_million = match model {
        m if m.contains("text-embedding-3-small") => 0.02,
        m if m.contains("text-embedding-3-large") => 0.13,
        m if m.contains("text-embedding-ada-002") => 0.10,
        _ => return None,
    };

    Some((tokens as f64 / 1_000_000.0) * price_per_million)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_cost() {
        // 1000 tokens of text-embedding-3-small = $0.00002
        let cost = calculate_cost("text-embedding-3-small", 1000);
        assert!(cost.is_some());
        assert!((cost.unwrap() - 0.00002).abs() < 0.0000001);

        // Unknown model returns None
        let cost = calculate_cost("unknown-model", 1000);
        assert!(cost.is_none());
    }
}
