use merco_llmproxy::config::{LlmConfig, Provider};
use merco_llmproxy::traits::{ChatMessage, CompletionRequest, LlmProvider, ChatMessageRole};
use std::env;
use std::error::Error;

#[tokio::main]
async fn main() {
    // Load API key from environment variable
    let api_key = env::var("OPENROUTER_API_KEY")
        .expect("API key is required.");

    // Configure for OpenRouter (using OpenAI provider)
    let config = LlmConfig::new(Provider::OpenAI)
        .with_api_key(api_key)
        .with_base_url("https://openrouter.ai/api/v1".to_string()); // Explicitly set OpenRouter URL

    println!("Using config: {:?}", config);

    // Get the provider
    let provider = merco_llmproxy::get_provider(config).expect("Failed to get provider");

    // Create a simple request
    let request = CompletionRequest {
        model: "openai/gpt-3.5-turbo".to_string(), // Added model field
        messages: vec![ChatMessage {
            role: ChatMessageRole::User, // Use imported ChatMessageRole
            content: Some("Say hello!".to_string()), // Fixed content type to Option<String>
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: Some(0.7),
        max_tokens: Some(50),
        tools: None,
    };

    println!("Sending request: {:?}", request);

    // Make the API call
    match provider.completion(request).await {
        Ok(response) => {
            println!("Success! Response: {:?}", response);
        }
        Err(e) => {
            eprintln!("Error during API call: {}", e);
            // Attempt to print underlying reqwest error if available
            if let Some(source) = e.source() {
                 if let Some(reqwest_err) = source.downcast_ref::<reqwest::Error>() {
                     eprintln!("Underlying Reqwest Error: {:?}", reqwest_err);
                     eprintln!("Is status error: {}", reqwest_err.is_status());
                     if let Some(status) = reqwest_err.status() {
                         eprintln!("Status code: {}", status);
                     }
                     if let Some(url) = reqwest_err.url() {
                        eprintln!("URL: {}", url);
                     }
                 } else {
                     eprintln!("Underlying error source: {:?}", source);
                 }
             }
        }
    }
} 