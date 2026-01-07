//! VoyageAI embeddings provider.

use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::errors::ClientError;
use crate::http::HttpClient;
use crate::models::{EmbedRequest, EmbedResponse, InputType, Usage};
use crate::providers::EmbeddingProvider;

const VOYAGEAI_API_URL: &str = "https://api.voyageai.com/v1/embeddings";

/// VoyageAI embeddings provider.
#[derive(Debug)]
pub struct VoyageAIProvider {
    api_key: String,
    http_client: HttpClient,
}

impl VoyageAIProvider {
    /// Create a new VoyageAI provider with the given API key.
    pub fn new(api_key: String, http_client: HttpClient) -> Self {
        Self {
            api_key,
            http_client,
        }
    }

    /// Create a new VoyageAI provider from environment variable.
    pub fn from_env(http_client: HttpClient) -> Result<Self, ClientError> {
        let api_key = std::env::var("VOYAGE_API_KEY").map_err(|_| ClientError::MissingApiKey {
            provider: "voyageai".to_string(),
        })?;
        Ok(Self::new(api_key, http_client))
    }
}

/// VoyageAI API request body.
#[derive(Debug, Serialize)]
struct VoyageAIEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    input_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimension: Option<u32>,
}

/// VoyageAI API response.
#[derive(Debug, Deserialize)]
struct VoyageAIEmbeddingResponse {
    data: Vec<VoyageAIEmbedding>,
    model: String,
    usage: VoyageAIUsage,
}

#[derive(Debug, Deserialize)]
struct VoyageAIEmbedding {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct VoyageAIUsage {
    total_tokens: u64,
}

/// VoyageAI API error response.
#[derive(Debug, Deserialize)]
struct VoyageAIErrorResponse {
    detail: String,
}

#[async_trait]
impl EmbeddingProvider for VoyageAIProvider {
    fn name(&self) -> &'static str {
        "voyageai"
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, ClientError> {
        debug!(
            model = %request.model,
            inputs = request.inputs.len(),
            "Sending VoyageAI embedding request"
        );

        let input_type_value = request.input_type;
        let input_type_str = input_type_value.map(|it| match it {
            InputType::Query => "query",
            InputType::Document => "document",
        });

        let input_count = request.inputs.len();
        let body = VoyageAIEmbeddingRequest {
            model: &request.model,
            input: &request.inputs,
            input_type: input_type_str,
            output_dimension: request.dimensions,
        };

        let body_json = serde_json::to_string(&body)?;
        // Use request's api_key if provided, otherwise use provider's api_key
        let api_key = request.api_key.as_ref().unwrap_or(&self.api_key).clone();

        let start = Instant::now();
        let response = self
            .http_client
            .send_with_retry(|client| {
                client
                    .post(VOYAGEAI_API_URL)
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
                serde_json::from_str::<VoyageAIErrorResponse>(&response_text)
            {
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

        let voyage_response: VoyageAIEmbeddingResponse = serde_json::from_str(&response_text)?;

        // Sort embeddings by index
        let mut embeddings: Vec<_> = voyage_response.data.into_iter().collect();
        embeddings.sort_by_key(|e| e.index);

        let embedding_vectors: Vec<Vec<f32>> =
            embeddings.into_iter().map(|e| e.embedding).collect();

        let dimensions = embedding_vectors.first().map(|e| e.len()).unwrap_or(0);

        // VoyageAI pricing: voyage-3 $0.06/1M, voyage-3-lite $0.02/1M
        let cost = calculate_cost(&request.model, voyage_response.usage.total_tokens);

        Ok(EmbedResponse {
            embeddings: embedding_vectors,
            model: voyage_response.model,
            provider: "voyageai".to_string(),
            dimensions,
            input_count,
            input_type: input_type_value,
            latency_ms,
            usage: Usage {
                tokens: voyage_response.usage.total_tokens,
                cost,
            },
        })
    }
}

fn calculate_cost(model: &str, tokens: u64) -> Option<f64> {
    let price_per_million = match model {
        m if m.contains("voyage-3-lite") => 0.02,
        m if m.contains("voyage-3") => 0.06,
        m if m.contains("voyage-code-3") => 0.18,
        m if m.contains("voyage-finance-2") => 0.12,
        m if m.contains("voyage-law-2") => 0.12,
        m if m.contains("voyage-2") => 0.10,
        _ => return None,
    };

    Some((tokens as f64 / 1_000_000.0) * price_per_million)
}
