//!
//! Provider Implementations Module
//! 
//! This module contains the concrete implementations for each supported LLM provider.
//! Each provider implements the `LlmProvider` trait defined in `crate::traits`.

// Declare provider implementation modules here
pub mod openai;
pub mod ollama;
// pub mod anthropic; // Example for future provider

// Re-export provider structs for easier access from the library root.
pub use openai::OpenAIProvider;
pub use ollama::OllamaProvider; 