#![warn(missing_docs)] // Enforce documentation coverage
//! A Rust library providing a unified interface for various Large Language Model (LLM) providers.
//! Inspired by LiteLLM, this crate aims to simplify interaction with different LLMs
//! through a common configuration and trait implementation.

pub mod config;
pub mod providers;
pub mod traits;
pub mod tools;

pub use config::{ConfigError, LlmConfig, Provider};
pub use providers::{OllamaProvider, OpenAIProvider};
pub use traits::{
    ChatMessage, CompletionKind, CompletionRequest, CompletionResponse, CompletionStream,
    CompletionStreamChunk, JsonSchema, LlmProvider, ProviderError, StreamContentDelta, Tool,
    ToolCallFunction, ToolCallRequest, ToolCallStreamDelta, TokenUsage,
};

// Re-export tool utilities 
pub use tools::{execute_tool, get_all_tools, get_tools_by_names, register_tool, ToolExecutor, ToolRegistry};

// Conditionally re-export the macro if the feature is enabled
#[cfg(feature = "macros")]
pub use tools::merco_tool;

// Optional: A factory function to create a provider instance based on config
use std::sync::Arc;

/// Creates a provider instance based on the provided configuration.
///
/// This function validates the configuration and returns a dynamic dispatch trait object (`Arc<dyn LlmProvider>`) 
/// allowing interaction with the selected provider through the common `LlmProvider` trait.
///
/// # Arguments
///
/// * `config` - The `LlmConfig` specifying the provider, model, credentials, etc.
///
/// # Errors
///
/// Returns `ProviderError::ConfigError` if the configuration is invalid for the selected provider.
/// Returns `ProviderError::Unsupported` if the selected provider is not yet implemented.
///
/// # Examples
///
/// ```no_run
/// use merco_llmproxy::{LlmConfig, Provider, get_provider, CompletionRequest, ChatMessage};
///
/// let config = LlmConfig::new(Provider::Ollama)
///     .with_base_url("http://localhost:11434".to_string()); // Optional, default assumed if Ollama
///
/// // Validate the config early (optional, but good practice)
/// config.validate().expect("Invalid config");
///
/// let provider = get_provider(config).expect("Failed to get provider");
///
/// // Example request (model specified here)
/// let request = CompletionRequest {
///     model: "qwen3:4b".to_string(),
///     messages: vec![ChatMessage::user("Why is the sky blue?".to_string())],
///     // ... other request options
///     ..Default::default()
/// };
///
/// // Now use the provider methods with the request...
/// // let response = provider.completion(request).await;
/// ```
pub fn get_provider(config: LlmConfig) -> Result<Arc<dyn LlmProvider>, ProviderError> {
    config.validate().map_err(|e| ProviderError::ConfigError(e.to_string()))?;

    match config.provider {
        Provider::OpenAI => Ok(Arc::new(OpenAIProvider::new(config))),
        Provider::Ollama => Ok(Arc::new(OllamaProvider::new(config))),
        Provider::Anthropic => Err(ProviderError::Unsupported("Anthropic provider not yet implemented".to_string())),
        Provider::Custom => Err(ProviderError::Unsupported("Custom provider logic not yet implemented".to_string())),
    }
}
