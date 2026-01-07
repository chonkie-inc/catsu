//! Google Gemini embeddings provider.

use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::errors::ClientError;
use crate::http::HttpClient;
use crate::models::{EmbedRequest, EmbedResponse, InputType, Usage};
use crate::providers::EmbeddingProvider;

const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Google Gemini embeddings provider.
#[derive(Debug)]
pub struct GeminiProvider {
    api_key: String,
    http_client: HttpClient,
}

impl GeminiProvider {
    /// Create a new Gemini provider with the given API key.
    pub fn new(api_key: String, http_client: HttpClient) -> Self {
        Self {
            api_key,
            http_client,
        }
    }

    /// Create a new Gemini provider from environment variable.
    pub fn from_env(http_client: HttpClient) -> Result<Self, ClientError> {
        let api_key = std::env::var("GOOGLE_API_KEY")
            .or_else(|_| std::env::var("GEMINI_API_KEY"))
            .map_err(|_| ClientError::MissingApiKey {
                provider: "gemini".to_string(),
            })?;
        Ok(Self::new(api_key, http_client))
    }
}

/// Gemini content for embedding request.
#[derive(Debug, Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Debug, Deserialize)]
struct GeminiEmbedding {
    values: Vec<f32>,
}

/// Gemini API error response.
#[derive(Debug, Deserialize)]
struct GeminiErrorResponse {
    error: GeminiError,
}

#[derive(Debug, Deserialize)]
struct GeminiError {
    message: String,
}

/// Single embed content request (for non-batch endpoint).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiSingleEmbedRequest {
    model: String,
    content: GeminiContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimensionality: Option<u32>,
}

/// Single embed response.
#[derive(Debug, Deserialize)]
struct GeminiSingleEmbedResponse {
    embedding: GeminiEmbedding,
}

#[async_trait]
impl EmbeddingProvider for GeminiProvider {
    fn name(&self) -> &'static str {
        "gemini"
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, ClientError> {
        debug!(
            model = %request.model,
            inputs = request.inputs.len(),
            "Sending Gemini embedding request"
        );

        let input_type_value = request.input_type;
        // Map to Gemini's taskType values
        let task_type = input_type_value.map(|it| match it {
            InputType::Query => "RETRIEVAL_QUERY".to_string(),
            InputType::Document => "RETRIEVAL_DOCUMENT".to_string(),
        });

        let input_count = request.inputs.len();
        // Use request's api_key if provided, otherwise use provider's api_key
        let api_key = request.api_key.as_ref().unwrap_or(&self.api_key).clone();
        let start = Instant::now();

        // Use single embedContent endpoint for each input
        // (batchEmbedContents requires additional API permissions)
        let url = format!(
            "{}/models/{}:embedContent",
            GEMINI_API_BASE, request.model
        );

        let mut embedding_vectors: Vec<Vec<f32>> = Vec::with_capacity(input_count);

        for text in &request.inputs {
            let body = GeminiSingleEmbedRequest {
                model: format!("models/{}", request.model),
                content: GeminiContent {
                    parts: vec![GeminiPart { text: text.clone() }],
                },
                task_type: task_type.clone(),
                output_dimensionality: request.dimensions,
            };
            let body_json = serde_json::to_string(&body)?;

            let response = self
                .http_client
                .send_with_retry(|client| {
                    client
                        .post(&url)
                        .header("x-goog-api-key", &api_key)
                        .header("Content-Type", "application/json")
                        .body(body_json.clone())
                })
                .await?;

            let status = response.status().as_u16();
            let response_text = response.text().await?;

            if status != 200 {
                if let Ok(error_response) =
                    serde_json::from_str::<GeminiErrorResponse>(&response_text)
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

            let gemini_response: GeminiSingleEmbedResponse =
                serde_json::from_str(&response_text)?;
            embedding_vectors.push(gemini_response.embedding.values);
        }

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let dimensions = embedding_vectors.first().map(|e| e.len()).unwrap_or(0);

        // Gemini doesn't return token count, estimate based on input
        // Rough estimate: 1 token per 5 characters
        let estimated_tokens: u64 = request
            .inputs
            .iter()
            .map(|s| (s.len() as u64 / 5).max(1))
            .sum();

        let cost = calculate_cost(&request.model, estimated_tokens);

        Ok(EmbedResponse {
            embeddings: embedding_vectors,
            model: request.model,
            provider: "gemini".to_string(),
            dimensions,
            input_count,
            input_type: input_type_value,
            latency_ms,
            usage: Usage {
                tokens: estimated_tokens,
                cost,
            },
        })
    }
}

fn calculate_cost(model: &str, tokens: u64) -> Option<f64> {
    // Gemini embedding pricing (as of 2024)
    let price_per_million = match model {
        m if m.contains("text-embedding") => 0.00, // Free tier
        m if m.contains("embedding-001") => 0.00,  // Free tier
        _ => return None,
    };

    Some((tokens as f64 / 1_000_000.0) * price_per_million)
}
