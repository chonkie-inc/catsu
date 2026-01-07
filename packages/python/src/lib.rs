//! Python bindings for catsu using PyO3.

use std::collections::HashMap;
use std::sync::Arc;

use once_cell::sync::Lazy;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3_async_runtimes::tokio::future_into_py;
use tokio::runtime::Runtime;

use catsu::{
    Client as RustClient, EmbedResponse as RustEmbedResponse, HttpConfig, InputType,
    ModelInfo as RustModelInfo,
};

/// Global tokio runtime for async operations.
static GLOBAL_RUNTIME: Lazy<Arc<Runtime>> = Lazy::new(|| {
    Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create global Tokio runtime"),
    )
});

/// Usage information from an embedding request.
#[pyclass]
#[derive(Clone)]
pub struct Usage {
    #[pyo3(get)]
    pub tokens: u64,
    #[pyo3(get)]
    pub cost: Option<f64>,
}

/// Information about an embedding model.
#[pyclass]
#[derive(Clone)]
pub struct ModelInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub provider: String,
    #[pyo3(get)]
    pub dimensions: u32,
    #[pyo3(get)]
    pub max_tokens: u32,
    #[pyo3(get)]
    pub supports_dimensions: bool,
    #[pyo3(get)]
    pub supports_input_type: bool,
    #[pyo3(get)]
    pub cost_per_million_tokens: Option<f64>,
}

impl From<RustModelInfo> for ModelInfo {
    fn from(info: RustModelInfo) -> Self {
        ModelInfo {
            name: info.name,
            provider: info.provider,
            dimensions: info.dimensions,
            max_tokens: info.max_tokens,
            supports_dimensions: info.supports_dimensions,
            supports_input_type: info.supports_input_type,
            cost_per_million_tokens: info.cost_per_million_tokens,
        }
    }
}

/// Response from an embedding request.
#[pyclass]
#[derive(Clone)]
pub struct EmbedResponse {
    #[pyo3(get)]
    pub embeddings: Vec<Vec<f32>>,
    #[pyo3(get)]
    pub model: String,
    #[pyo3(get)]
    pub provider: String,
    #[pyo3(get)]
    pub dimensions: usize,
    #[pyo3(get)]
    pub input_count: usize,
    #[pyo3(get)]
    pub input_type: Option<String>,
    #[pyo3(get)]
    pub latency_ms: f64,
    #[pyo3(get)]
    pub usage: Usage,
}

#[pymethods]
impl EmbedResponse {
    /// Convert embeddings to a numpy array.
    ///
    /// Returns:
    ///     A 2D numpy array of shape (input_count, dimensions)
    ///
    /// Example:
    ///     >>> response = client.embed("openai:text-embedding-3-small", ["hello", "world"])
    ///     >>> arr = response.to_numpy()
    ///     >>> arr.shape  # (2, 1536)
    pub fn to_numpy(&self, py: Python<'_>) -> PyResult<PyObject> {
        let numpy = py.import("numpy")?;
        let array = numpy.call_method1("array", (&self.embeddings,))?;
        Ok(array.into())
    }
}

impl From<RustEmbedResponse> for EmbedResponse {
    fn from(response: RustEmbedResponse) -> Self {
        let input_type = response.input_type.map(|it| match it {
            InputType::Query => "query".to_string(),
            InputType::Document => "document".to_string(),
        });

        EmbedResponse {
            embeddings: response.embeddings,
            model: response.model,
            provider: response.provider,
            dimensions: response.dimensions,
            input_count: response.input_count,
            input_type,
            latency_ms: response.latency_ms,
            usage: Usage {
                tokens: response.usage.tokens,
                cost: response.usage.cost,
            },
        }
    }
}

/// High-performance embeddings client.
///
/// Example:
///     >>> from catsu import Client
///     >>> client = Client()
///     >>> response = client.embed("openai:text-embedding-3-small", "Hello, world!")
///     >>> print(len(response.embeddings[0]))  # 1536
#[pyclass(name = "Client")]
pub struct CatsuClient {
    inner: Arc<RustClient>,
}

#[pymethods]
impl CatsuClient {
    /// Create a new Client.
    ///
    /// API keys are read from environment variables:
    /// - OPENAI_API_KEY for OpenAI
    /// - VOYAGE_API_KEY for VoyageAI
    /// - COHERE_API_KEY for Cohere
    /// - etc.
    ///
    /// Args:
    ///     api_keys: Optional dict of provider names to API keys
    ///     max_retries: Maximum number of retry attempts (default: 3)
    ///     timeout: Request timeout in seconds (default: 30)
    #[new]
    #[pyo3(signature = (api_keys=None, max_retries=None, timeout=None))]
    pub fn new(
        api_keys: Option<HashMap<String, String>>,
        max_retries: Option<u32>,
        timeout: Option<u64>,
    ) -> PyResult<Self> {
        let mut config = HttpConfig::default();
        if let Some(retries) = max_retries {
            config.max_retries = retries;
        }
        if let Some(secs) = timeout {
            config.timeout_secs = secs;
        }

        let inner = if let Some(keys) = api_keys {
            RustClient::with_api_keys_and_config(keys, config)
        } else {
            RustClient::with_config(config)
        }
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Generate embeddings for input text(s).
    ///
    /// Args:
    ///     model: Model name (with or without "provider:" prefix)
    ///     input: Single text string or list of text strings to embed
    ///     provider: Optional explicit provider name (e.g., "openai", "voyageai")
    ///     input_type: Optional input type hint ("query" or "document")
    ///     dimensions: Optional output dimensions (if model supports it)
    ///     api_key: Optional API key override for this request
    ///
    /// Returns:
    ///     EmbedResponse with embeddings and usage information
    ///
    /// Example:
    ///     >>> response = client.embed("openai:text-embedding-3-small", "hello world")
    ///     >>> response = client.embed("text-embedding-3-small", "hello", provider="openai")
    ///     >>> response = client.embed("voyageai:voyage-3", ["hello", "world"])
    ///     >>> response = client.embed("openai:text-embedding-3-small", "hello", api_key="sk-...")
    #[pyo3(signature = (model, input, provider=None, input_type=None, dimensions=None, api_key=None))]
    pub fn embed(
        &self,
        py: Python<'_>,
        model: &str,
        input: &Bound<'_, PyAny>,
        provider: Option<&str>,
        input_type: Option<&str>,
        dimensions: Option<u32>,
        api_key: Option<String>,
    ) -> PyResult<EmbedResponse> {
        // Accept both string and list of strings
        let inputs: Vec<String> = if let Ok(s) = input.downcast::<PyString>() {
            vec![s.to_string()]
        } else if let Ok(list) = input.extract::<Vec<String>>() {
            list
        } else {
            return Err(PyRuntimeError::new_err(
                "input must be a string or list of strings",
            ));
        };

        let model = model.to_string();
        let provider_owned = provider.map(|s| s.to_string());
        let input_type_enum = match input_type {
            Some("query") => Some(InputType::Query),
            Some("document") => Some(InputType::Document),
            Some(other) => {
                return Err(PyRuntimeError::new_err(format!(
                    "Invalid input_type '{}'. Must be 'query' or 'document'",
                    other
                )))
            }
            None => None,
        };

        let rt = GLOBAL_RUNTIME.clone();
        let client = &self.inner;

        // Release GIL while doing async work
        let result = py.allow_threads(move || {
            rt.block_on(async {
                client
                    .embed_with_api_key(
                        &model,
                        inputs,
                        input_type_enum,
                        dimensions,
                        provider_owned.as_deref(),
                        api_key,
                    )
                    .await
            })
        });

        result
            .map(EmbedResponse::from)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Async version of embed(). Generate embeddings for input text(s) asynchronously.
    ///
    /// Args:
    ///     model: Model name (with or without "provider:" prefix)
    ///     input: Single text string or list of text strings to embed
    ///     provider: Optional explicit provider name (e.g., "openai", "voyageai")
    ///     input_type: Optional input type hint ("query" or "document")
    ///     dimensions: Optional output dimensions (if model supports it)
    ///     api_key: Optional API key override for this request
    ///
    /// Returns:
    ///     EmbedResponse with embeddings and usage information
    ///
    /// Example:
    ///     >>> import asyncio
    ///     >>> async def main():
    ///     ...     client = Client()
    ///     ...     response = await client.aembed("openai:text-embedding-3-small", "hello")
    ///     >>> asyncio.run(main())
    #[pyo3(signature = (model, input, provider=None, input_type=None, dimensions=None, api_key=None))]
    pub fn aembed<'py>(
        &self,
        py: Python<'py>,
        model: &str,
        input: &Bound<'_, PyAny>,
        provider: Option<&str>,
        input_type: Option<&str>,
        dimensions: Option<u32>,
        api_key: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Accept both string and list of strings
        let inputs: Vec<String> = if let Ok(s) = input.downcast::<PyString>() {
            vec![s.to_string()]
        } else if let Ok(list) = input.extract::<Vec<String>>() {
            list
        } else {
            return Err(PyRuntimeError::new_err(
                "input must be a string or list of strings",
            ));
        };

        let model = model.to_string();
        let provider_owned = provider.map(|s| s.to_string());
        let input_type_enum = match input_type {
            Some("query") => Some(InputType::Query),
            Some("document") => Some(InputType::Document),
            Some(other) => {
                return Err(PyRuntimeError::new_err(format!(
                    "Invalid input_type '{}'. Must be 'query' or 'document'",
                    other
                )))
            }
            None => None,
        };

        let client = Arc::clone(&self.inner);

        future_into_py(py, async move {
            let result = client
                .embed_with_api_key(
                    &model,
                    inputs,
                    input_type_enum,
                    dimensions,
                    provider_owned.as_deref(),
                    api_key,
                )
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            Ok(EmbedResponse::from(result))
        })
    }

    /// List available providers.
    ///
    /// Returns:
    ///     List of provider names that have API keys configured
    pub fn list_providers(&self) -> Vec<String> {
        self.inner
            .list_providers()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Check if a provider is available.
    ///
    /// Args:
    ///     provider: Provider name (e.g., "openai", "voyageai")
    ///
    /// Returns:
    ///     True if the provider has an API key configured
    pub fn has_provider(&self, provider: &str) -> bool {
        self.inner.has_provider(provider)
    }

    /// List available models from the catalog.
    ///
    /// Args:
    ///     provider: Optional provider name to filter by (e.g., "openai", "voyageai")
    ///
    /// Returns:
    ///     List of ModelInfo objects with model metadata
    ///
    /// Example:
    ///     >>> models = client.list_models()  # All models
    ///     >>> models = client.list_models("openai")  # Only OpenAI models
    ///     >>> for m in models:
    ///     ...     print(f"{m.provider}:{m.name} - {m.dimensions} dims")
    #[pyo3(signature = (provider=None))]
    pub fn list_models(&self, provider: Option<&str>) -> Vec<ModelInfo> {
        self.inner
            .list_models(provider)
            .into_iter()
            .map(ModelInfo::from)
            .collect()
    }

    /// Context manager entry - returns self.
    ///
    /// Example:
    ///     >>> with Client() as client:
    ///     ...     response = client.embed("openai:text-embedding-3-small", "hello")
    pub fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Context manager exit - no cleanup needed.
    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    pub fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        // No cleanup needed - return False to not suppress exceptions
        false
    }

    /// Async context manager entry - returns self.
    ///
    /// Example:
    ///     >>> async with Client() as client:
    ///     ...     response = await client.aembed("openai:text-embedding-3-small", "hello")
    pub fn __aenter__<'py>(slf: Py<Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        future_into_py(py, async move { Ok(slf) })
    }

    /// Async context manager exit - no cleanup needed.
    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    pub fn __aexit__<'py>(
        &self,
        py: Python<'py>,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // No cleanup needed - return False to not suppress exceptions
        future_into_py(py, async move { Ok(false) })
    }
}

/// Python module definition.
#[pymodule]
fn _catsu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CatsuClient>()?;
    m.add_class::<EmbedResponse>()?;
    m.add_class::<Usage>()?;
    m.add_class::<ModelInfo>()?;
    Ok(())
}
