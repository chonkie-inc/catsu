//! Mistral AI embeddings provider.

use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::errors::ClientError;
use crate::http::HttpClient;
use crate::models::{EmbedRequest, EmbedResponse, Usage};
use crate::providers::EmbeddingProvider;

const MISTRAL_API_URL: &str = "https://api.mistral.ai/v1/embeddings";

/// Mistral AI embeddings provider.
#[derive(Debug)]
pub struct MistralProvider {
    api_key: String,
    http_client: HttpClient,
}

impl MistralProvider {
    /// Create a new Mistral provider with the given API key.
    pub fn new(api_key: String, http_client: HttpClient) -> Self {
        Self {
            api_key,
            http_client,
        }
    }

    /// Create a new Mistral provider from environment variable.
    pub fn from_env(http_client: HttpClient) -> Result<Self, ClientError> {
        let api_key = std::env::var("MISTRAL_API_KEY").map_err(|_| ClientError::MissingApiKey {
            provider: "mistral".to_string(),
        })?;
        Ok(Self::new(api_key, http_client))
    }
}

/// Mistral API request body.
#[derive(Debug, Serialize)]
struct MistralEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<&'a str>,
}

/// Mistral API response.
#[derive(Debug, Deserialize)]
struct MistralEmbeddingResponse {
    data: Vec<MistralEmbedding>,
    model: String,
    usage: MistralUsage,
}

#[derive(Debug, Deserialize)]
struct MistralEmbedding {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct MistralUsage {
    total_tokens: u64,
}

/// Mistral API error response.
#[derive(Debug, Deserialize)]
struct MistralErrorResponse {
    message: String,
}

#[async_trait]
impl EmbeddingProvider for MistralProvider {
    fn name(&self) -> &'static str {
        "mistral"
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, ClientError> {
        debug!(
            model = %request.model,
            inputs = request.inputs.len(),
            "Sending Mistral embedding request"
        );

        // Mistral doesn't use input_type, but we preserve it for the response
        let input_type_value = request.input_type;
        if input_type_value.is_some() {
            debug!("Mistral doesn't use input_type parameter, ignoring");
        }

        let input_count = request.inputs.len();
        let body = MistralEmbeddingRequest {
            model: &request.model,
            input: &request.inputs,
            encoding_format: Some("float"),
        };

        let body_json = serde_json::to_string(&body)?;
        // Use request's api_key if provided, otherwise use provider's api_key
        let api_key = request.api_key.as_ref().unwrap_or(&self.api_key).clone();

        let start = Instant::now();
        let response = self
            .http_client
            .send_with_retry(|client| {
                client
                    .post(MISTRAL_API_URL)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("Content-Type", "application/json")
                    .body(body_json.clone())
            })
            .await?;

        let status = response.status().as_u16();
        let response_text = response.text().await?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        if status != 200 {
            if let Ok(error_response) = serde_json::from_str::<MistralErrorResponse>(&response_text)
            {
                return Err(ClientError::Api {
                    status,
                    message: error_response.message,
                });
            }
            return Err(ClientError::Api {
                status,
                message: response_text,
            });
        }

        let mistral_response: MistralEmbeddingResponse = serde_json::from_str(&response_text)?;

        // Sort embeddings by index
        let mut embeddings: Vec<_> = mistral_response.data.into_iter().collect();
        embeddings.sort_by_key(|e| e.index);

        let embedding_vectors: Vec<Vec<f32>> =
            embeddings.into_iter().map(|e| e.embedding).collect();

        let dimensions = embedding_vectors.first().map(|e| e.len()).unwrap_or(0);
        let total_tokens = mistral_response.usage.total_tokens;

        let cost = calculate_cost(&request.model, total_tokens);

        Ok(EmbedResponse {
            embeddings: embedding_vectors,
            model: mistral_response.model,
            provider: "mistral".to_string(),
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
    let price_per_million = match model {
        m if m.contains("mistral-embed") => 0.10,
        _ => return None,
    };

    Some((tokens as f64 / 1_000_000.0) * price_per_million)
}
