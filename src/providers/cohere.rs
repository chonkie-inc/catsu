//! Cohere embeddings provider.

use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::errors::ClientError;
use crate::http::HttpClient;
use crate::models::{EmbedRequest, EmbedResponse, InputType, Usage};
use crate::providers::EmbeddingProvider;

const COHERE_API_URL: &str = "https://api.cohere.com/v1/embed";

/// Cohere embeddings provider.
#[derive(Debug)]
pub struct CohereProvider {
    api_key: String,
    http_client: HttpClient,
}

impl CohereProvider {
    /// Create a new Cohere provider with the given API key.
    pub fn new(api_key: String, http_client: HttpClient) -> Self {
        Self {
            api_key,
            http_client,
        }
    }

    /// Create a new Cohere provider from environment variable.
    pub fn from_env(http_client: HttpClient) -> Result<Self, ClientError> {
        let api_key = std::env::var("COHERE_API_KEY").map_err(|_| ClientError::MissingApiKey {
            provider: "cohere".to_string(),
        })?;
        Ok(Self::new(api_key, http_client))
    }
}

/// Cohere API request body.
#[derive(Debug, Serialize)]
struct CohereEmbeddingRequest<'a> {
    model: &'a str,
    texts: &'a [String],
    embedding_types: Vec<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_type: Option<&'a str>,
}

/// Cohere API response.
#[derive(Debug, Deserialize)]
struct CohereEmbeddingResponse {
    embeddings: CohereEmbeddings,
    meta: CohereMeta,
}

#[derive(Debug, Deserialize)]
struct CohereEmbeddings {
    float: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct CohereMeta {
    billed_units: CohereBilledUnits,
}

#[derive(Debug, Deserialize)]
struct CohereBilledUnits {
    input_tokens: u64,
}

/// Cohere API error response.
#[derive(Debug, Deserialize)]
struct CohereErrorResponse {
    message: String,
}

#[async_trait]
impl EmbeddingProvider for CohereProvider {
    fn name(&self) -> &'static str {
        "cohere"
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, ClientError> {
        debug!(
            model = %request.model,
            inputs = request.inputs.len(),
            "Sending Cohere embedding request"
        );

        let input_type_value = request.input_type;
        // Map to Cohere's input_type values (v3 models require input_type)
        let cohere_input_type = Some(match input_type_value {
            Some(InputType::Query) => "search_query",
            Some(InputType::Document) | None => "search_document",
        });

        let input_count = request.inputs.len();
        let body = CohereEmbeddingRequest {
            model: &request.model,
            texts: &request.inputs,
            embedding_types: vec!["float"],
            input_type: cohere_input_type,
        };

        let body_json = serde_json::to_string(&body)?;
        // Use request's api_key if provided, otherwise use provider's api_key
        let api_key = request.api_key.as_ref().unwrap_or(&self.api_key).clone();

        let start = Instant::now();
        let response = self
            .http_client
            .send_with_retry(|client| {
                client
                    .post(COHERE_API_URL)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("Content-Type", "application/json")
                    .body(body_json.clone())
            })
            .await?;

        let status = response.status().as_u16();
        let response_text = response.text().await?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        if status != 200 {
            if let Ok(error_response) = serde_json::from_str::<CohereErrorResponse>(&response_text)
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

        let cohere_response: CohereEmbeddingResponse = serde_json::from_str(&response_text)?;

        let embedding_vectors = cohere_response.embeddings.float;
        let dimensions = embedding_vectors.first().map(|e| e.len()).unwrap_or(0);
        let total_tokens = cohere_response.meta.billed_units.input_tokens;

        let cost = calculate_cost(&request.model, total_tokens);

        Ok(EmbedResponse {
            embeddings: embedding_vectors,
            model: request.model,
            provider: "cohere".to_string(),
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
        m if m.contains("embed-english-v3") => 0.10,
        m if m.contains("embed-multilingual-v3") => 0.10,
        m if m.contains("embed-english-light-v3") => 0.10,
        m if m.contains("embed-multilingual-light-v3") => 0.10,
        _ => return None,
    };

    Some((tokens as f64 / 1_000_000.0) * price_per_million)
}
