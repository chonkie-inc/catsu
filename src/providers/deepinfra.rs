//! DeepInfra embeddings provider.

use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::errors::ClientError;
use crate::http::HttpClient;
use crate::models::{EmbedRequest, EmbedResponse, Usage};
use crate::providers::EmbeddingProvider;

const DEEPINFRA_API_URL: &str = "https://api.deepinfra.com/v1/openai/embeddings";

/// DeepInfra embeddings provider.
#[derive(Debug)]
pub struct DeepInfraProvider {
    api_key: String,
    http_client: HttpClient,
}

impl DeepInfraProvider {
    /// Create a new DeepInfra provider with the given API key.
    pub fn new(api_key: String, http_client: HttpClient) -> Self {
        Self {
            api_key,
            http_client,
        }
    }

    /// Create a new DeepInfra provider from environment variable.
    pub fn from_env(http_client: HttpClient) -> Result<Self, ClientError> {
        let api_key =
            std::env::var("DEEPINFRA_API_KEY").map_err(|_| ClientError::MissingApiKey {
                provider: "deepinfra".to_string(),
            })?;
        Ok(Self::new(api_key, http_client))
    }
}

/// DeepInfra API request body (OpenAI-compatible).
#[derive(Debug, Serialize)]
struct DeepInfraEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

/// DeepInfra API response.
#[derive(Debug, Deserialize)]
struct DeepInfraEmbeddingResponse {
    data: Vec<DeepInfraEmbedding>,
    model: String,
    usage: DeepInfraUsage,
}

#[derive(Debug, Deserialize)]
struct DeepInfraEmbedding {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct DeepInfraUsage {
    total_tokens: u64,
}

/// DeepInfra API error response.
#[derive(Debug, Deserialize)]
struct DeepInfraErrorResponse {
    error: DeepInfraError,
}

#[derive(Debug, Deserialize)]
struct DeepInfraError {
    message: String,
}

#[async_trait]
impl EmbeddingProvider for DeepInfraProvider {
    fn name(&self) -> &'static str {
        "deepinfra"
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, ClientError> {
        debug!(
            model = %request.model,
            inputs = request.inputs.len(),
            "Sending DeepInfra embedding request"
        );

        // DeepInfra doesn't support input_type
        let input_type_value = request.input_type;
        if input_type_value.is_some() {
            debug!("DeepInfra doesn't use input_type parameter, ignoring");
        }

        let input_count = request.inputs.len();
        let body = DeepInfraEmbeddingRequest {
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
                    .post(DEEPINFRA_API_URL)
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
                serde_json::from_str::<DeepInfraErrorResponse>(&response_text)
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

        let di_response: DeepInfraEmbeddingResponse = serde_json::from_str(&response_text)?;

        // Sort embeddings by index
        let mut embeddings: Vec<_> = di_response.data.into_iter().collect();
        embeddings.sort_by_key(|e| e.index);

        let embedding_vectors: Vec<Vec<f32>> =
            embeddings.into_iter().map(|e| e.embedding).collect();

        let dimensions = embedding_vectors.first().map(|e| e.len()).unwrap_or(0);
        let total_tokens = di_response.usage.total_tokens;

        let cost = calculate_cost(&request.model, total_tokens);

        Ok(EmbedResponse {
            embeddings: embedding_vectors,
            model: di_response.model,
            provider: "deepinfra".to_string(),
            dimensions,
            input_count,
            input_type: input_type_value,
            latency_ms,
            usage: Usage { tokens: total_tokens, cost },
        })
    }
}

fn calculate_cost(model: &str, tokens: u64) -> Option<f64> {
    // DeepInfra embedding pricing (per 1M tokens)
    let price_per_million = match model {
        m if m.contains("bge") => 0.005,
        m if m.contains("e5") => 0.005,
        m if m.contains("gte") => 0.005,
        m if m.contains("Qwen") => 0.01,
        _ => 0.005, // Default
    };

    Some((tokens as f64 / 1_000_000.0) * price_per_million)
}
