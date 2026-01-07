//! Error types for catsu client.

use thiserror::Error;

/// Main error type for catsu operations.
#[derive(Error, Debug)]
pub enum ClientError {
    /// HTTP request failed
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON serialization/deserialization failed
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// API returned an error response
    #[error("API error: {message} (status: {status})")]
    Api { status: u16, message: String },

    /// Rate limited by the API
    #[error("Rate limited: retry after {retry_after:?} seconds")]
    RateLimited { retry_after: Option<u64> },

    /// Authentication failed
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    /// Invalid input provided
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Model not found
    #[error("Model not found: {model}")]
    ModelNotFound { model: String },

    /// Provider not found
    #[error("Provider not found: {provider}")]
    ProviderNotFound { provider: String },

    /// Max retries exceeded
    #[error("Max retries exceeded after {attempts} attempts")]
    MaxRetriesExceeded { attempts: u32 },

    /// Request timeout
    #[error("Request timed out")]
    Timeout,

    /// Missing API key
    #[error("Missing API key for provider: {provider}")]
    MissingApiKey { provider: String },
}

impl ClientError {
    /// Check if this error is retryable.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            ClientError::RateLimited { .. } => true,
            ClientError::Timeout => true,
            ClientError::Api { status, .. } => is_retryable_status(*status),
            _ => false,
        }
    }
}

/// Check if an HTTP status code is retryable.
#[must_use]
pub fn is_retryable_status(status: u16) -> bool {
    matches!(status, 429 | 408 | 409 | 500 | 502 | 503 | 504)
}
