//! HTTP client with retry and exponential backoff.

use std::time::Duration;

use rand::Rng;
use reqwest::{Client, RequestBuilder, Response};
use tracing::warn;

use crate::errors::{is_retryable_status, ClientError};

/// Configuration for HTTP requests.
#[derive(Debug, Clone)]
pub struct HttpConfig {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Initial backoff duration in milliseconds.
    pub initial_backoff_ms: u64,
    /// Maximum backoff duration in milliseconds.
    pub max_backoff_ms: u64,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 100,
            max_backoff_ms: 30_000,
            timeout_secs: 30,
        }
    }
}

/// HTTP client wrapper with retry logic.
#[derive(Debug, Clone)]
pub struct HttpClient {
    client: Client,
    config: HttpConfig,
}

impl HttpClient {
    /// Create a new HTTP client with the given configuration.
    pub fn new(config: HttpConfig) -> Result<Self, ClientError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .pool_max_idle_per_host(10)
            .build()?;

        Ok(Self { client, config })
    }

    /// Get the underlying reqwest client.
    pub fn inner(&self) -> &Client {
        &self.client
    }

    /// Send a request with retry logic.
    ///
    /// The request builder factory is called for each attempt since
    /// RequestBuilder cannot be cloned after body is set.
    pub async fn send_with_retry<F>(&self, request_factory: F) -> Result<Response, ClientError>
    where
        F: Fn(&Client) -> RequestBuilder,
    {
        let mut attempts = 0;
        let mut backoff_ms = self.config.initial_backoff_ms;

        loop {
            attempts += 1;
            let request = request_factory(&self.client);

            match request.send().await {
                Ok(response) => {
                    let status = response.status().as_u16();

                    if response.status().is_success() {
                        return Ok(response);
                    }

                    // Check if retryable
                    if is_retryable_status(status) && attempts <= self.config.max_retries {
                        // Extract retry-after header if present
                        let retry_after = response
                            .headers()
                            .get("retry-after")
                            .and_then(|v| v.to_str().ok())
                            .and_then(|v| v.parse::<u64>().ok());

                        let wait_ms = if let Some(secs) = retry_after {
                            secs * 1000
                        } else {
                            self.calculate_backoff(backoff_ms)
                        };

                        warn!(
                            status = status,
                            attempt = attempts,
                            wait_ms = wait_ms,
                            "Retryable error, waiting before retry"
                        );

                        tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                        backoff_ms = (backoff_ms * 2).min(self.config.max_backoff_ms);
                        continue;
                    }

                    // Non-retryable error or max retries exceeded
                    if status == 429 {
                        return Err(ClientError::RateLimited { retry_after: None });
                    }

                    // Try to extract error message from response body
                    let message = response
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string());

                    return Err(ClientError::Api { status, message });
                }
                Err(e) => {
                    // Network error
                    if attempts <= self.config.max_retries {
                        let wait_ms = self.calculate_backoff(backoff_ms);

                        warn!(
                            error = %e,
                            attempt = attempts,
                            wait_ms = wait_ms,
                            "Network error, waiting before retry"
                        );

                        tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                        backoff_ms = (backoff_ms * 2).min(self.config.max_backoff_ms);
                        continue;
                    }

                    if e.is_timeout() {
                        return Err(ClientError::Timeout);
                    }

                    return Err(ClientError::Http(e));
                }
            }
        }
    }

    /// Calculate backoff with jitter.
    fn calculate_backoff(&self, base_ms: u64) -> u64 {
        let mut rng = rand::rng();
        let jitter: f64 = rng.random();
        let jitter_ms = (base_ms as f64 * jitter) as u64;
        base_ms + jitter_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = HttpConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.initial_backoff_ms, 100);
        assert_eq!(config.timeout_secs, 30);
    }

    #[test]
    fn test_is_retryable() {
        assert!(is_retryable_status(429));
        assert!(is_retryable_status(500));
        assert!(is_retryable_status(503));
        assert!(!is_retryable_status(400));
        assert!(!is_retryable_status(401));
        assert!(!is_retryable_status(404));
    }
}
