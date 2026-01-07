//! Nomic embeddings provider.

use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::errors::ClientError;
use crate::http::HttpClient;
use crate::models::{EmbedRequest, EmbedResponse, InputType, Usage};
use crate::providers::EmbeddingProvider;

const NOMIC_API_URL: &str = "https://api-atlas.nomic.ai/v1/embedding/text";

/// Nomic embeddings provider.
#[derive(Debug)]
pub struct NomicProvider {
    api_key: String,
    http_client: HttpClient,
}

impl NomicProvider {
    /// Create a new Nomic provider with the given API key.
    pub fn new(api_key: String, http_client: HttpClient) -> Self {
        Self {
            api_key,
            http_client,
        }
    }

    /// Create a new Nomic provider from environment variable.
    pub fn from_env(http_client: HttpClient) -> Result<Self, ClientError> {
        let api_key = std::env::var("NOMIC_API_KEY").map_err(|_| ClientError::MissingApiKey {
            provider: "nomic".to_string(),
        })?;
        Ok(Self::new(api_key, http_client))
    }
}

/// Nomic API request body.
#[derive(Debug, Serialize)]
struct NomicEmbeddingRequest<'a> {
    model: &'a str,
    texts: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    task_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensionality: Option<u32>,
    long_text_mode: &'a str,
    max_tokens_per_text: u32,
}

/// Nomic API response.
#[derive(Debug, Deserialize)]
struct NomicEmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
    model: String,
    usage: NomicUsage,
}

#[derive(Debug, Deserialize)]
struct NomicUsage {
    total_tokens: u64,
}

/// Nomic API error response.
#[derive(Debug, Deserialize)]
struct NomicErrorResponse {
    detail: String,
}

#[async_trait]
impl EmbeddingProvider for NomicProvider {
    fn name(&self) -> &'static str {
        "nomic"
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, ClientError> {
        debug!(
            model = %request.model,
            inputs = request.inputs.len(),
            "Sending Nomic embedding request"
        );

        let input_type_value = request.input_type;
        // Map to Nomic's task_type values
        let task_type = input_type_value.map(|it| match it {
            InputType::Query => "search_query",
            InputType::Document => "search_document",
        });

        // Dimensionality only supported on v1.5
        let dimensionality = if request.model.contains("v1.5") {
            request.dimensions
        } else {
            if request.dimensions.is_some() {
                debug!("Dimensions only supported on nomic-embed-text-v1.5, ignoring");
            }
            None
        };

        let input_count = request.inputs.len();
        let body = NomicEmbeddingRequest {
            model: &request.model,
            texts: &request.inputs,
            task_type,
            dimensionality,
            long_text_mode: "mean",
            max_tokens_per_text: 8192,
        };

        let body_json = serde_json::to_string(&body)?;
        // Use request's api_key if provided, otherwise use provider's api_key
        let api_key = request.api_key.as_ref().unwrap_or(&self.api_key).clone();

        let start = Instant::now();
        let response = self
            .http_client
            .send_with_retry(|client| {
                client
                    .post(NOMIC_API_URL)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("Content-Type", "application/json")
                    .body(body_json.clone())
            })
            .await?;

        let status = response.status().as_u16();
        let response_text = response.text().await?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        if status != 200 {
            if let Ok(error_response) = serde_json::from_str::<NomicErrorResponse>(&response_text) {
                return Err(ClientError::Api {
                    status,
                    message: error_response.detail,
                });
            }
            return Err(ClientError::Api {
                status,
                message: response_text,
            });
        }

        let nomic_response: NomicEmbeddingResponse = serde_json::from_str(&response_text)?;

        let embedding_vectors = nomic_response.embeddings;
        let dimensions = embedding_vectors.first().map(|e| e.len()).unwrap_or(0);
        let total_tokens = nomic_response.usage.total_tokens;

        let cost = calculate_cost(&request.model, total_tokens);

        Ok(EmbedResponse {
            embeddings: embedding_vectors,
            model: nomic_response.model,
            provider: "nomic".to_string(),
            dimensions,
            input_count,
            input_type: input_type_value,
            latency_ms,
            usage: Usage { tokens: total_tokens, cost },
        })
    }
}

fn calculate_cost(model: &str, tokens: u64) -> Option<f64> {
    let price_per_million = match model {
        m if m.contains("nomic-embed-text") => 0.10,
        _ => return None,
    };

    Some((tokens as f64 / 1_000_000.0) * price_per_million)
}
