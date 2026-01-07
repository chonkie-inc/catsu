//! Cloudflare Workers AI embeddings provider.

use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::errors::ClientError;
use crate::http::HttpClient;
use crate::models::{EmbedRequest, EmbedResponse, Usage};
use crate::providers::EmbeddingProvider;

const CLOUDFLARE_API_BASE: &str = "https://api.cloudflare.com/client/v4/accounts";

/// Cloudflare Workers AI embeddings provider.
#[derive(Debug)]
pub struct CloudflareProvider {
    api_key: String,
    account_id: String,
    http_client: HttpClient,
}

impl CloudflareProvider {
    /// Create a new Cloudflare provider with the given API key and account ID.
    pub fn new(api_key: String, account_id: String, http_client: HttpClient) -> Self {
        Self {
            api_key,
            account_id,
            http_client,
        }
    }

    /// Create a new Cloudflare provider from environment variables.
    pub fn from_env(http_client: HttpClient) -> Result<Self, ClientError> {
        let api_key =
            std::env::var("CLOUDFLARE_API_TOKEN").map_err(|_| ClientError::MissingApiKey {
                provider: "cloudflare".to_string(),
            })?;
        let account_id =
            std::env::var("CLOUDFLARE_ACCOUNT_ID").map_err(|_| ClientError::MissingApiKey {
                provider: "cloudflare (account_id)".to_string(),
            })?;
        Ok(Self::new(api_key, account_id, http_client))
    }
}

/// Cloudflare API request body.
#[derive(Debug, Serialize)]
struct CloudflareEmbeddingRequest<'a> {
    text: &'a [String],
}

/// Cloudflare API response.
#[derive(Debug, Deserialize)]
struct CloudflareEmbeddingResponse {
    result: CloudflareResult,
    success: bool,
}

#[derive(Debug, Deserialize)]
struct CloudflareResult {
    data: Vec<Vec<f32>>,
    #[serde(default)]
    shape: Vec<usize>,
}

/// Cloudflare API error response.
#[derive(Debug, Deserialize)]
struct CloudflareErrorResponse {
    errors: Vec<CloudflareError>,
}

#[derive(Debug, Deserialize)]
struct CloudflareError {
    message: String,
}

#[async_trait]
impl EmbeddingProvider for CloudflareProvider {
    fn name(&self) -> &'static str {
        "cloudflare"
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, ClientError> {
        debug!(
            model = %request.model,
            inputs = request.inputs.len(),
            "Sending Cloudflare embedding request"
        );

        // Cloudflare doesn't support input_type or dimensions
        let input_type_value = request.input_type;
        if input_type_value.is_some() {
            debug!("Cloudflare doesn't use input_type parameter, ignoring");
        }
        if request.dimensions.is_some() {
            debug!("Cloudflare doesn't support custom dimensions, ignoring");
        }

        let input_count = request.inputs.len();
        let body = CloudflareEmbeddingRequest {
            text: &request.inputs,
        };

        let body_json = serde_json::to_string(&body)?;
        // Use request's api_key if provided, otherwise use provider's api_key
        let api_key = request.api_key.as_ref().unwrap_or(&self.api_key).clone();

        let url = format!(
            "{}/{}/ai/run/{}",
            CLOUDFLARE_API_BASE, self.account_id, request.model
        );

        let start = Instant::now();
        let response = self
            .http_client
            .send_with_retry(|client| {
                client
                    .post(&url)
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
                serde_json::from_str::<CloudflareErrorResponse>(&response_text)
            {
                let message = error_response
                    .errors
                    .first()
                    .map(|e| e.message.clone())
                    .unwrap_or_else(|| "Unknown error".to_string());
                return Err(ClientError::Api { status, message });
            }
            return Err(ClientError::Api {
                status,
                message: response_text,
            });
        }

        let cf_response: CloudflareEmbeddingResponse = serde_json::from_str(&response_text)?;

        if !cf_response.success {
            return Err(ClientError::Api {
                status,
                message: "Cloudflare API returned success=false".to_string(),
            });
        }

        let embedding_vectors = cf_response.result.data;
        let dimensions = cf_response
            .result
            .shape
            .get(1)
            .copied()
            .unwrap_or_else(|| embedding_vectors.first().map(|e| e.len()).unwrap_or(0));

        // Cloudflare doesn't return token count, estimate based on input
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
            provider: "cloudflare".to_string(),
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

fn calculate_cost(_model: &str, _tokens: u64) -> Option<f64> {
    // Cloudflare Workers AI has free tier, pricing varies
    Some(0.0)
}
