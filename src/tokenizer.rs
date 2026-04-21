use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use once_cell::sync::Lazy;
use tokie::Tokenizer;
use tracing::warn;

use crate::catalog::get_model;

static TOKENIZER_CACHE: Lazy<Mutex<HashMap<String, Arc<Tokenizer>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub fn count_tokens_for_model(provider: &str, model: &str, texts: &[String]) -> u64 {
    let tokenizer_info = match get_model(provider, model) {
        Some(info) => match info.tokenizer {
            Some(t) => t,
            None => return fallback_count(texts),
        },
        None => return fallback_count(texts),
    };

    let repo_name = match tokenizer_info.engine.as_str() {
        "huggingface" => tokenizer_info.name.clone(),
        "tiktoken" => match tiktoken_to_repo(&tokenizer_info.name) {
            Some(repo) => repo.to_string(),
            None => return fallback_count(texts),
        },
        _ => return fallback_count(texts),
    };

    match get_or_load_tokenizer(&repo_name) {
        Some(tokenizer) => {
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            tokenizer.count_tokens_batch(&refs).iter().sum::<usize>() as u64
        }
        None => fallback_count(texts),
    }
}

fn get_or_load_tokenizer(name: &str) -> Option<Arc<Tokenizer>> {
    let mut cache = TOKENIZER_CACHE.lock().ok()?;

    if let Some(tok) = cache.get(name) {
        return Some(Arc::clone(tok));
    }

    match Tokenizer::from_pretrained(name) {
        Ok(tok) => {
            let tok = Arc::new(tok);
            cache.insert(name.to_string(), Arc::clone(&tok));
            Some(tok)
        }
        Err(e) => {
            warn!(
                tokenizer = name,
                error = %e,
                "Failed to load tokenizer, falling back to estimate"
            );
            None
        }
    }
}

fn tiktoken_to_repo(name: &str) -> Option<&'static str> {
    match name {
        "cl100k_base" => Some("xenova/gpt-4"),
        "o200k_base" => Some("xenova/gpt-4o"),
        _ => None,
    }
}

pub fn fallback_count(texts: &[String]) -> u64 {
    match get_or_load_tokenizer("xenova/gpt-4") {
        Some(tokenizer) => {
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            tokenizer.count_tokens_batch(&refs).iter().sum::<usize>() as u64
        }
        None => texts.iter().map(|s| (s.len() as u64 / 5).max(1)).sum(),
    }
}
