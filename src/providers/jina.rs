//! Jina AI embeddings provider.

use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::errors::ClientError;
use crate::http::HttpClient;
use crate::models::{EmbedRequest, EmbedResponse, InputType, Usage};
use crate::providers::EmbeddingProvider;

const JINA_API_URL: &str = "https://api.jina.ai/v1/embeddings";

/// Jina AI embeddings provider.
#[derive(Debug)]
pub struct JinaProvider {
    api_key: String,
    http_client: HttpClient,
}

impl JinaProvider {
    /// Create a new Jina provider with the given API key.
    pub fn new(api_key: String, http_client: HttpClient) -> Self {
        Self {
            api_key,
            http_client,
        }
    }

    /// Create a new Jina provider from environment variable.
    pub fn from_env(http_client: HttpClient) -> Result<Self, ClientError> {
        let api_key = std::env::var("JINA_API_KEY").map_err(|_| ClientError::MissingApiKey {
            provider: "jina".to_string(),
        })?;
        Ok(Self::new(api_key, http_client))
    }
}

/// Jina API request body.
#[derive(Debug, Serialize)]
struct JinaEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    task: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
    normalized: bool,
}

/// Jina API response.
#[derive(Debug, Deserialize)]
struct JinaEmbeddingResponse {
    data: Vec<JinaEmbedding>,
    model: String,
    usage: JinaUsage,
}

#[derive(Debug, Deserialize)]
struct JinaEmbedding {
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct JinaUsage {
    total_tokens: u64,
}

/// Jina API error response.
#[derive(Debug, Deserialize)]
struct JinaErrorResponse {
    detail: String,
}

#[async_trait]
impl EmbeddingProvider for JinaProvider {
    fn name(&self) -> &'static str {
        "jina"
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, ClientError> {
        debug!(
            model = %request.model,
            inputs = request.inputs.len(),
            "Sending Jina embedding request"
        );

        let input_type_value = request.input_type;
        // Map to Jina's task values
        let task = input_type_value.map(|it| match it {
            InputType::Query => "retrieval.query",
            InputType::Document => "retrieval.passage",
        });

        let input_count = request.inputs.len();
        let body = JinaEmbeddingRequest {
            model: &request.model,
            input: &request.inputs,
            task,
            dimensions: request.dimensions,
            normalized: true,
        };

        let body_json = serde_json::to_string(&body)?;
        // Use request's api_key if provided, otherwise use provider's api_key
        let api_key = request.api_key.as_ref().unwrap_or(&self.api_key).clone();

        let start = Instant::now();
        let response = self
            .http_client
            .send_with_retry(|client| {
                client
                    .post(JINA_API_URL)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("Content-Type", "application/json")
                    .body(body_json.clone())
            })
            .await?;

        let status = response.status().as_u16();
        let response_text = response.text().await?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        if status != 200 {
            if let Ok(error_response) = serde_json::from_str::<JinaErrorResponse>(&response_text) {
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

        let jina_response: JinaEmbeddingResponse = serde_json::from_str(&response_text)?;

        let embedding_vectors: Vec<Vec<f32>> = jina_response
            .data
            .into_iter()
            .map(|e| e.embedding)
            .collect();

        let dimensions = embedding_vectors.first().map(|e| e.len()).unwrap_or(0);
        let total_tokens = jina_response.usage.total_tokens;

        let cost = calculate_cost(&request.model, total_tokens);

        Ok(EmbedResponse {
            embeddings: embedding_vectors,
            model: jina_response.model,
            provider: "jina".to_string(),
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
        m if m.contains("jina-embeddings-v3") => 0.02,
        m if m.contains("jina-embeddings-v2") => 0.02,
        m if m.contains("jina-colbert-v2") => 0.02,
        _ => return None,
    };

    Some((tokens as f64 / 1_000_000.0) * price_per_million)
}
