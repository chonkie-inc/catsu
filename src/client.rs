//! Main catsu client.

use std::collections::HashMap;
use std::sync::Arc;

use crate::catalog::find_model_by_name;
use crate::errors::ClientError;
use crate::http::{HttpClient, HttpConfig};
use crate::models::{EmbedRequest, EmbedResponse, InputType};
use crate::providers::{
    CloudflareProvider, CohereProvider, DeepInfraProvider, EmbeddingProvider, GeminiProvider,
    JinaProvider, MistralProvider, MixedbreadProvider, NomicProvider, OpenAIProvider,
    TogetherProvider, VoyageAIProvider,
};

/// Macro to register a provider from environment if API key is available.
macro_rules! register_provider_from_env {
    ($providers:expr, $http_client:expr, $($name:literal => $provider:ty),+ $(,)?) => {
        $(
            if let Ok(provider) = <$provider>::from_env($http_client.clone()) {
                $providers.insert($name.to_string(), Arc::new(provider) as Arc<dyn EmbeddingProvider>);
            }
        )+
    };
}

/// Macro to register a provider with explicit API key.
macro_rules! register_provider_with_key {
    ($providers:expr, $api_keys:expr, $http_client:expr, $($name:literal => $provider:ty),+ $(,)?) => {
        $(
            if let Some(api_key) = $api_keys.get($name) {
                let provider = <$provider>::new(api_key.clone(), $http_client.clone());
                $providers.insert($name.to_string(), Arc::new(provider) as Arc<dyn EmbeddingProvider>);
            }
        )+
    };
}

/// Main catsu client for generating embeddings.
///
/// # Example
///
/// ```no_run
/// use catsu::Client;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let client = Client::new()?;
///     let response = client.embed(
///         "openai:text-embedding-3-small",
///         vec!["Hello, world!".to_string()],
///     ).await?;
///     println!("Embeddings: {:?}", response.embeddings);
///     Ok(())
/// }
/// ```
pub struct Client {
    providers: HashMap<String, Arc<dyn EmbeddingProvider>>,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("providers", &self.providers.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl Client {
    /// Create a new catsu client with default configuration.
    ///
    /// API keys are read from environment variables:
    /// - `OPENAI_API_KEY` for OpenAI
    /// - `VOYAGE_API_KEY` for VoyageAI
    /// - `COHERE_API_KEY` for Cohere
    /// - `JINA_API_KEY` for Jina
    /// - `MISTRAL_API_KEY` for Mistral
    /// - `GOOGLE_API_KEY` or `GEMINI_API_KEY` for Gemini
    /// - `TOGETHER_API_KEY` for Together
    /// - `MIXEDBREAD_API_KEY` for Mixedbread
    /// - `NOMIC_API_KEY` for Nomic
    /// - `DEEPINFRA_API_KEY` for DeepInfra
    /// - `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` for Cloudflare
    pub fn new() -> Result<Self, ClientError> {
        Self::with_config(HttpConfig::default())
    }

    /// Create a new catsu client with custom HTTP configuration.
    pub fn with_config(config: HttpConfig) -> Result<Self, ClientError> {
        let http_client = HttpClient::new(config)?;
        let mut providers: HashMap<String, Arc<dyn EmbeddingProvider>> = HashMap::new();

        register_provider_from_env!(providers, http_client,
            "openai" => OpenAIProvider,
            "voyageai" => VoyageAIProvider,
            "cohere" => CohereProvider,
            "jina" => JinaProvider,
            "mistral" => MistralProvider,
            "gemini" => GeminiProvider,
            "together" => TogetherProvider,
            "mixedbread" => MixedbreadProvider,
            "nomic" => NomicProvider,
            "deepinfra" => DeepInfraProvider,
        );

        // Cloudflare requires both API key and account ID
        if let Ok(provider) = CloudflareProvider::from_env(http_client) {
            providers.insert(
                "cloudflare".to_string(),
                Arc::new(provider) as Arc<dyn EmbeddingProvider>,
            );
        }

        Ok(Self { providers })
    }

    /// Create a new catsu client with explicit API keys.
    ///
    /// # Arguments
    ///
    /// * `api_keys` - Map of provider names to API keys.
    ///   For Cloudflare, provide both "cloudflare" (API token) and "cloudflare_account_id".
    pub fn with_api_keys(api_keys: HashMap<String, String>) -> Result<Self, ClientError> {
        Self::with_api_keys_and_config(api_keys, HttpConfig::default())
    }

    /// Create a new catsu client with explicit API keys and custom HTTP configuration.
    ///
    /// # Arguments
    ///
    /// * `api_keys` - Map of provider names to API keys.
    ///   For Cloudflare, provide both "cloudflare" (API token) and "cloudflare_account_id".
    /// * `config` - HTTP configuration (timeout, max_retries, etc.)
    pub fn with_api_keys_and_config(
        api_keys: HashMap<String, String>,
        config: HttpConfig,
    ) -> Result<Self, ClientError> {
        let http_client = HttpClient::new(config)?;
        let mut providers: HashMap<String, Arc<dyn EmbeddingProvider>> = HashMap::new();

        register_provider_with_key!(providers, api_keys, http_client,
            "openai" => OpenAIProvider,
            "voyageai" => VoyageAIProvider,
            "cohere" => CohereProvider,
            "jina" => JinaProvider,
            "mistral" => MistralProvider,
            "gemini" => GeminiProvider,
            "together" => TogetherProvider,
            "mixedbread" => MixedbreadProvider,
            "nomic" => NomicProvider,
            "deepinfra" => DeepInfraProvider,
        );

        // Cloudflare requires both API key and account ID
        if let (Some(api_key), Some(account_id)) = (
            api_keys.get("cloudflare"),
            api_keys.get("cloudflare_account_id"),
        ) {
            let provider =
                CloudflareProvider::new(api_key.clone(), account_id.clone(), http_client);
            providers.insert(
                "cloudflare".to_string(),
                Arc::new(provider) as Arc<dyn EmbeddingProvider>,
            );
        }

        Ok(Self { providers })
    }

    /// Generate embeddings for the given inputs.
    ///
    /// # Arguments
    ///
    /// * `model` - Model name in format "provider:model" (e.g., "openai:text-embedding-3-small")
    /// * `inputs` - List of text strings to embed
    ///
    /// # Returns
    ///
    /// `EmbedResponse` containing the embeddings and usage information.
    pub async fn embed(
        &self,
        model: &str,
        inputs: Vec<String>,
    ) -> Result<EmbedResponse, ClientError> {
        self.embed_with_options(model, inputs, None, None).await
    }

    /// Generate embeddings with additional options.
    ///
    /// # Arguments
    ///
    /// * `model` - Model name (with or without "provider:" prefix)
    /// * `inputs` - List of text strings to embed
    /// * `input_type` - Optional input type hint (query or document)
    /// * `dimensions` - Optional output dimensions (if model supports it)
    /// * `provider` - Optional explicit provider name (if not included in model string)
    pub async fn embed_with_options(
        &self,
        model: &str,
        inputs: Vec<String>,
        input_type: Option<InputType>,
        dimensions: Option<u32>,
    ) -> Result<EmbedResponse, ClientError> {
        self.embed_full(model, inputs, input_type, dimensions, None)
            .await
    }

    /// Generate embeddings with all options including explicit provider.
    ///
    /// # Arguments
    ///
    /// * `model` - Model name (with or without "provider:" prefix)
    /// * `inputs` - List of text strings to embed
    /// * `input_type` - Optional input type hint (query or document)
    /// * `dimensions` - Optional output dimensions (if model supports it)
    /// * `provider` - Optional explicit provider name (overrides provider in model string)
    /// * `api_key` - Optional API key override for this request
    pub async fn embed_full(
        &self,
        model: &str,
        inputs: Vec<String>,
        input_type: Option<InputType>,
        dimensions: Option<u32>,
        provider: Option<&str>,
    ) -> Result<EmbedResponse, ClientError> {
        self.embed_with_api_key(model, inputs, input_type, dimensions, provider, None)
            .await
    }

    /// Generate embeddings with all options including API key override.
    ///
    /// # Arguments
    ///
    /// * `model` - Model name (with or without "provider:" prefix)
    /// * `inputs` - List of text strings to embed
    /// * `input_type` - Optional input type hint (query or document)
    /// * `dimensions` - Optional output dimensions (if model supports it)
    /// * `provider` - Optional explicit provider name (overrides provider in model string)
    /// * `api_key` - Optional API key override for this request
    pub async fn embed_with_api_key(
        &self,
        model: &str,
        inputs: Vec<String>,
        input_type: Option<InputType>,
        dimensions: Option<u32>,
        provider: Option<&str>,
        api_key: Option<String>,
    ) -> Result<EmbedResponse, ClientError> {
        let (provider_name, model_name) = if let Some(p) = provider {
            // Explicit provider provided - use model as-is (strip provider prefix if present)
            let model_name = model
                .split_once(':')
                .map(|(_, m)| m.to_string())
                .unwrap_or_else(|| model.to_string());
            (p.to_string(), model_name)
        } else {
            self.parse_model_string(model)?
        };

        let provider_impl =
            self.providers
                .get(&provider_name)
                .ok_or_else(|| ClientError::ProviderNotFound {
                    provider: provider_name.clone(),
                })?;

        let request = EmbedRequest {
            model: model_name,
            inputs,
            input_type,
            dimensions,
            api_key,
        };

        provider_impl.embed(request).await
    }

    /// Parse model string into provider and model name.
    ///
    /// Supports formats:
    /// - "provider:model" (e.g., "openai:text-embedding-3-small")
    /// - "model" (auto-detects provider from catalog)
    fn parse_model_string(&self, model: &str) -> Result<(String, String), ClientError> {
        if let Some((provider, model_name)) = model.split_once(':') {
            Ok((provider.to_string(), model_name.to_string()))
        } else {
            // Try to auto-detect provider from the model catalog
            if let Some(model_info) = find_model_by_name(model) {
                Ok((model_info.provider, model.to_string()))
            } else {
                // Model not found in catalog - return error with helpful message
                Err(ClientError::InvalidInput(format!(
                    "Model '{}' not found in catalog. Use 'provider:model' format (e.g., 'openai:text-embedding-3-small') or check available models with list_models()",
                    model
                )))
            }
        }
    }

    /// List available providers.
    #[must_use]
    pub fn list_providers(&self) -> Vec<&str> {
        self.providers.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a provider is available.
    #[must_use]
    pub fn has_provider(&self, provider: &str) -> bool {
        self.providers.contains_key(provider)
    }

    /// List available models, optionally filtered by provider.
    ///
    /// # Arguments
    ///
    /// * `provider` - Optional provider name to filter by (e.g., "openai", "voyageai")
    ///
    /// # Returns
    ///
    /// List of `ModelInfo` objects from the catalog.
    #[must_use]
    pub fn list_models(&self, provider: Option<&str>) -> Vec<crate::models::ModelInfo> {
        crate::catalog::list_models(provider)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_string() {
        let client = Client::with_api_keys(HashMap::new()).unwrap();

        // With provider prefix
        let (provider, model) = client
            .parse_model_string("openai:text-embedding-3-small")
            .unwrap();
        assert_eq!(provider, "openai");
        assert_eq!(model, "text-embedding-3-small");

        // Without provider prefix - should auto-detect from catalog
        let (provider, model) = client
            .parse_model_string("text-embedding-3-small")
            .unwrap();
        assert_eq!(provider, "openai");
        assert_eq!(model, "text-embedding-3-small");

        // VoyageAI model without prefix - should auto-detect
        let (provider, model) = client.parse_model_string("voyage-3-large").unwrap();
        assert_eq!(provider, "voyageai");
        assert_eq!(model, "voyage-3-large");

        // Unknown model without prefix - should error
        let result = client.parse_model_string("non-existent-model-xyz");
        assert!(result.is_err());
    }
}
