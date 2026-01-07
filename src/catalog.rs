//! Model catalog for catsu.
//!
//! This module provides access to model metadata embedded at compile time.

use std::collections::HashMap;

use once_cell::sync::Lazy;
use serde::Deserialize;

use crate::models::ModelInfo;

/// Raw JSON model data embedded at compile time.
const MODELS_JSON: &str = include_str!("data/models.json");

/// Model entry from the JSON catalog (matches the JSON structure).
#[derive(Debug, Deserialize)]
struct CatalogEntry {
    name: String,
    provider: String,
    dimensions: u32,
    max_input_tokens: u32,
    cost_per_million_tokens: Option<f64>,
    #[serde(default)]
    supports_dimensions: bool,
    #[serde(default)]
    supports_input_type: bool,
}

impl From<CatalogEntry> for ModelInfo {
    fn from(entry: CatalogEntry) -> Self {
        ModelInfo {
            name: entry.name,
            provider: entry.provider,
            dimensions: entry.dimensions,
            max_tokens: entry.max_input_tokens,
            supports_dimensions: entry.supports_dimensions,
            supports_input_type: entry.supports_input_type,
            cost_per_million_tokens: entry.cost_per_million_tokens,
        }
    }
}

/// Lazily parsed model catalog.
static CATALOG: Lazy<HashMap<String, Vec<ModelInfo>>> = Lazy::new(|| {
    let raw: HashMap<String, Vec<CatalogEntry>> =
        serde_json::from_str(MODELS_JSON).expect("Failed to parse embedded models.json");

    raw.into_iter()
        .map(|(provider, entries)| {
            let models: Vec<ModelInfo> = entries.into_iter().map(ModelInfo::from).collect();
            (provider, models)
        })
        .collect()
});

/// Get all models, optionally filtered by provider.
///
/// # Arguments
///
/// * `provider` - Optional provider name to filter by (e.g., "openai", "voyageai")
///
/// # Returns
///
/// List of `ModelInfo` objects matching the filter.
pub fn list_models(provider: Option<&str>) -> Vec<ModelInfo> {
    match provider {
        Some(p) => CATALOG.get(p).cloned().unwrap_or_default(),
        None => CATALOG.values().flatten().cloned().collect(),
    }
}

/// Get model info by provider and model name.
pub fn get_model(provider: &str, model: &str) -> Option<ModelInfo> {
    CATALOG
        .get(provider)?
        .iter()
        .find(|m| m.name == model)
        .cloned()
}

/// List all available providers in the catalog.
pub fn list_catalog_providers() -> Vec<&'static str> {
    CATALOG.keys().map(|s| s.as_str()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_all_models() {
        let models = list_models(None);
        assert!(!models.is_empty());
    }

    #[test]
    fn test_list_models_by_provider() {
        let models = list_models(Some("openai"));
        assert!(!models.is_empty());
        assert!(models.iter().all(|m| m.provider == "openai"));
    }

    #[test]
    fn test_get_model() {
        let model = get_model("openai", "text-embedding-3-small");
        assert!(model.is_some());
        let model = model.unwrap();
        assert_eq!(model.name, "text-embedding-3-small");
        assert_eq!(model.provider, "openai");
    }
}
