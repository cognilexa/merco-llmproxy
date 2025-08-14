use thiserror::Error;

/// APP site URL
pub const APP_SITE_URL: &str = "merco.app";

/// APP site name
pub const APP_SITE_NAME: &str = "Merco LLM";

/// Represents the supported LLM providers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Provider {
    /// OpenAI models (via OpenAI API or compatible endpoints like OpenRouter).
    OpenAI,
    /// Ollama local models.
    Ollama,
    /// Anthropic Claude models.
    Anthropic,
    /// Placeholder for custom or self-hosted models using a specific base URL.
    Custom, 
}

/// Configuration for initializing an LLM provider.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// The specific provider to use.
    pub provider: Provider,
    /// The API key required by the provider (if any).
    pub api_key: Option<String>,
    /// The base URL for the provider's API endpoint.
    /// Optional, mainly for `Custom` providers or overriding defaults (e.g., OpenRouter).
    pub base_url: Option<String>,
}

/// Errors that can occur during configuration validation.
#[derive(Error, Debug)]
pub enum ConfigError {
    /// Missing API key required for the specified provider.
    #[error("Missing API key for provider: {0:?}")]
    MissingApiKey(Provider),
    /// Missing base URL required for the `Custom` provider.
    #[error("Missing base URL for custom provider")]
    MissingBaseUrl,
}

impl LlmConfig {
    /// Creates a new basic configuration.
    pub fn new(provider: Provider) -> Self {
        LlmConfig {
            provider,
            api_key: None,
            base_url: None,
        }
    }

    /// Sets the API key for the configuration (builder style).
    pub fn with_api_key(mut self, api_key: String) -> Self {
        self.api_key = Some(api_key);
        self
    }

    /// Sets the base URL for the configuration (builder style).
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = Some(base_url);
        self
    }

    /// Validates the configuration based on the selected provider's requirements.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if validation fails (e.g., missing API key).
    pub fn validate(&self) -> Result<(), ConfigError> {
        match self.provider {
            Provider::OpenAI | Provider::Anthropic => {
                if self.api_key.is_none() {
                    return Err(ConfigError::MissingApiKey(self.provider.clone()));
                }
            }
            Provider::Custom => {
                if self.base_url.is_none() {
                    return Err(ConfigError::MissingBaseUrl);
                }
                // Note: Custom provider might still optionally use an API key,
                // but we don't enforce it here.
            }
            Provider::Ollama => {
                // Ollama typically doesn't require an API key.
                // Base URL defaults to localhost if not provided.
            }
        }
        Ok(())
    }
} 