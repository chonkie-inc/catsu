//! Mixedbread AI embeddings provider.

use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::errors::ClientError;
use crate::http::HttpClient;
use crate::models::{EmbedRequest, EmbedResponse, InputType, Usage};
use crate::providers::EmbeddingProvider;

const MIXEDBREAD_API_URL: &str = "https://api.mixedbread.com/v1/embeddings";

/// Mixedbread AI embeddings provider.
#[derive(Debug)]
pub struct MixedbreadProvider {
    api_key: String,
    http_client: HttpClient,
}

impl MixedbreadProvider {
    /// Create a new Mixedbread provider with the given API key.
    pub fn new(api_key: String, http_client: HttpClient) -> Self {
        Self {
            api_key,
            http_client,
        }
    }

    /// Create a new Mixedbread provider from environment variable.
    pub fn from_env(http_client: HttpClient) -> Result<Self, ClientError> {
        let api_key =
            std::env::var("MIXEDBREAD_API_KEY").map_err(|_| ClientError::MissingApiKey {
                provider: "mixedbread".to_string(),
            })?;
        Ok(Self::new(api_key, http_client))
    }
}

/// Mixedbread API request body.
#[derive(Debug, Serialize)]
struct MixedbreadEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
    normalized: bool,
    encoding_format: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt: Option<&'a str>,
}

/// Mixedbread API response.
#[derive(Debug, Deserialize)]
struct MixedbreadEmbeddingResponse {
    data: Vec<MixedbreadEmbedding>,
    model: String,
    usage: MixedbreadUsage,
    #[serde(default)]
    dimensions: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct MixedbreadEmbedding {
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct MixedbreadUsage {
    total_tokens: u64,
}

/// Mixedbread API error response.
#[derive(Debug, Deserialize)]
struct MixedbreadErrorResponse {
    message: String,
}

#[async_trait]
impl EmbeddingProvider for MixedbreadProvider {
    fn name(&self) -> &'static str {
        "mixedbread"
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, ClientError> {
        debug!(
            model = %request.model,
            inputs = request.inputs.len(),
            "Sending Mixedbread embedding request"
        );

        let input_type_value = request.input_type;
        // Map input_type to prompt for Mixedbread
        let prompt = input_type_value.map(|it| match it {
            InputType::Query => "Represent this sentence for searching relevant passages: ",
            InputType::Document => "",
        });
        let prompt = prompt.filter(|p| !p.is_empty());

        let input_count = request.inputs.len();
        let body = MixedbreadEmbeddingRequest {
            model: &request.model,
            input: &request.inputs,
            dimensions: request.dimensions,
            normalized: true,
            encoding_format: "float",
            prompt,
        };

        let body_json = serde_json::to_string(&body)?;
        // Use request's api_key if provided, otherwise use provider's api_key
        let api_key = request.api_key.as_ref().unwrap_or(&self.api_key).clone();

        let start = Instant::now();
        let response = self
            .http_client
            .send_with_retry(|client| {
                client
                    .post(MIXEDBREAD_API_URL)
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
                serde_json::from_str::<MixedbreadErrorResponse>(&response_text)
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

        let mxbai_response: MixedbreadEmbeddingResponse = serde_json::from_str(&response_text)?;

        let embedding_vectors: Vec<Vec<f32>> = mxbai_response
            .data
            .into_iter()
            .map(|e| e.embedding)
            .collect();

        let dimensions = mxbai_response
            .dimensions
            .unwrap_or_else(|| embedding_vectors.first().map(|e| e.len()).unwrap_or(0));
        let total_tokens = mxbai_response.usage.total_tokens;

        let cost = calculate_cost(&request.model, total_tokens);

        Ok(EmbedResponse {
            embeddings: embedding_vectors,
            model: mxbai_response.model,
            provider: "mixedbread".to_string(),
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
        m if m.contains("mxbai-embed-large") => 0.05,
        m if m.contains("mxbai-embed") => 0.02,
        _ => return None,
    };

    Some((tokens as f64 / 1_000_000.0) * price_per_million)
}
