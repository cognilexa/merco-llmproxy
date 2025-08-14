# merco-llmproxy

A Rust library providing a unified interface for various Large Language Model (LLM) providers, inspired by [LiteLLM](https://github.com/BerriAI/litellm).

This crate aims to simplify interaction with different LLMs by offering:

*   A common configuration structure (`LlmConfig`).
*   A unified asynchronous trait (`LlmProvider`) for chat completions.
*   Support for multiple providers (currently OpenAI-compatible APIs and Ollama).
*   Basic support for non-streaming tool calls (function calling).
*   A convenient macro to register Rust functions as LLM tools.

## Current Status

*   **Providers:** OpenAI (including proxies like OpenRouter), Ollama.
*   **Features:** Non-streaming Chat Completion, Non-streaming Tool Calls.
*   **Limitations:** Streaming Tool Calls are currently **not** supported reliably due to SSE parsing complexities and Ollama's JSON mode limitations.

## Installation

Add this crate to your `Cargo.toml` dependencies:

```toml
[dependencies]
merco-llmproxy = { git = "<your-repo-url>" } # Or path = "..." if local
# Required peer dependencies (ensure versions are compatible)
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["json"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
async-trait = "0.1"
futures = "0.3"
thiserror = "1.0"
```

*(Replace `<your-repo-url>` with the actual repository URL once published.)*

## Usage

### 1. Configuration

Create an `LlmConfig` specifying the provider, and any necessary credentials or URLs. The model is specified per-request.

```rust
use merco_llmproxy::{LlmConfig, Provider};

// Example: Configure for a local Ollama model
let ollama_config = LlmConfig::new(Provider::Ollama)
    .with_base_url("http://localhost:11434".to_string()); // Optional: Defaults if not set

// Example: Configure for OpenAI via OpenRouter (requires API key)
let openrouter_api_key = std::env::var("OPENROUTER_API_KEY")
    .expect("OPENROUTER_API_KEY must be set");

let openrouter_config = LlmConfig::new(Provider::OpenAI) // Use OpenAI provider type for compatible APIs
    .with_base_url("https://openrouter.ai/api/v1".to_string())
    .with_api_key(openrouter_api_key);

// Validate the config (optional, but recommended)
ollama_config.validate().expect("Invalid Ollama config");
openrouter_config.validate().expect("Invalid OpenRouter config");
```

**Environment Variables:**

*   For providers requiring API keys (like OpenAI/OpenRouter), ensure the corresponding key is set (e.g., `OPENROUTER_API_KEY`).

### 2. Get Provider Instance

Use the `get_provider` function to obtain a trait object (`Arc<dyn LlmProvider>`) based on the configuration.

```rust
# use merco_llmproxy::{LlmConfig, Provider, get_provider};
# let config = LlmConfig::new(Provider::Ollama);
let provider = match get_provider(config) {
    Ok(p) => p,
    Err(e) => {
        eprintln!("Failed to get provider: {}", e);
        // Handle error appropriately
        panic!(); 
    }
};
```

### 3. Simple Chat Completion (Non-streaming)

Create a `CompletionRequest` and call the `completion` method.

```rust
# use merco_llmproxy::{LlmConfig, Provider, get_provider};
# use merco_llmproxy::traits::{ChatMessage, CompletionRequest, CompletionKind};
# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
# let config = LlmConfig::new(Provider::Ollama);
# let provider = get_provider(config)?;
let request = CompletionRequest {
    model: "qwen3:4b".to_string(), // Specify the model HERE
    messages: vec![
        ChatMessage {
            role: "system".to_string(),
            content: Some("You are a helpful assistant.".to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: Some("Why is the sky blue?".to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
    ],
    temperature: Some(0.7),
    max_tokens: Some(100),
    tools: None, // No tools needed for this request
};

match provider.completion(request).await {
    Ok(response) => {
        println!("Finish Reason: {:?}", response.finish_reason);
        if let Some(usage) = response.usage {
            println!("Usage: Prompt={}, Completion={}, Total={}", 
                usage.prompt_tokens, usage.completion_tokens, usage.total_tokens);
        }
        match response.kind {
            CompletionKind::Message { content } => {
                println!("Response Content:\n{}", content);
            }
            CompletionKind::ToolCall { tool_calls } => {
                println!("Unexpected tool call requested: {:?}", tool_calls);
            }
        }
    }
    Err(e) => {
        eprintln!("Completion Error: {}", e);
    }
}
# Ok(())
# }
```

### 4. Defining and Using Tools with `#[merco_tool]`

The `merco_tool` attribute macro provides a convenient way to make your standard Rust functions callable by LLMs. When you annotate a function, it's automatically registered in a global tool registry.

**Step 1: Define Your Tools**

```rust
use merco_llmproxy::merco_tool;

/// Calculates the sum of two integers.
#[merco_tool(description = "Calculates the sum of two integers")]
fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}

/// Provides the weather forecast for a specific city.
#[merco_tool(description = "Provides the weather forecast for a specific city (e.g., London, Paris, Tokyo)")]
fn get_weather(city: String) -> String {
    match city.to_lowercase().as_str() {
        "london" => "Cloudy, 15°C".to_string(),
        "paris" => "Sunny, 22°C".to_string(),
        _ => format!("Sorry, weather for {} is unavailable.", city),
    }
}

/// Calculates the area of a circle (might not be needed for every request).
#[merco_tool(description = "Calculates the area of a circle given its radius")]
fn calculate_circle_area(radius: f64) -> f64 {
    std::f64::consts::PI * radius * radius
}

// ... define other tools as needed ...
```

**Step 2: Select Tools for the Request**

Instead of sending *all* registered tools every time, use `get_tools_by_names` to select only the relevant ones for the current task. This is more efficient and less confusing for the LLM.

```rust
# // Dummy imports and setup for example to compile
# use merco_llmproxy::{get_tools_by_names, execute_tool, ChatMessage, CompletionKind, CompletionRequest, LlmConfig, Provider, get_provider, Tool};
# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
#     let config = LlmConfig::new(Provider::Ollama);
#     let provider = get_provider(config)?;
#     // Assume add_numbers and get_weather were defined with #[merco_tool]

    // Define which tools are relevant for this specific request
    let relevant_tool_names = ["add_numbers", "get_weather"];
    let selected_tools: Vec<Tool> = get_tools_by_names(&relevant_tool_names);

    if selected_tools.len() != relevant_tool_names.len() {
        println!("Warning: Some requested tools were not found in the registry.");
    }

    // Create a completion request using ONLY the selected tools
    let request = CompletionRequest {
        model: "some-model".to_string(), // Replace with your actual model
        messages: vec![
            ChatMessage {
                role: "user".to_string(),
                content: Some("What is 15 plus 9, and what is the weather in Paris?".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        temperature: Some(0.1),
        max_tokens: Some(300),
        tools: Some(selected_tools), // Use the specifically selected tools
    };
#    // Dummy response handling
#    println!("Simulating request with tools: {:?}", request.tools.unwrap().iter().map(|t| &t.name).collect::<Vec<_>>());
#    let tool_calls = vec![ 
#       merco_llmproxy::traits::ToolCallRequest { id: "call_1".to_string(), function: merco_llmproxy::traits::ToolCallFunction { name: "add_numbers".to_string(), arguments: "{\"a\": 15, \"b\": 9}".to_string() } },
#       merco_llmproxy::traits::ToolCallRequest { id: "call_2".to_string(), function: merco_llmproxy::traits::ToolCallFunction { name: "get_weather".to_string(), arguments: "{\"city\": \"Paris\"}".to_string() } },
#    ];
#    let response_kind = CompletionKind::ToolCall { tool_calls }; 
#    // End dummy setup
    
    // Make the request to the LLM (using the `provider` instance)
    // let response = provider.completion(request).await?;
    
    // Handle tool calls (same as before)
    // match response.kind {
    match response_kind { // Using dummy response_kind for example
        CompletionKind::ToolCall { tool_calls } => {
            println!("\nTool Calls Requested:");
            for call in tool_calls {
                println!("  Tool: {}, Args: {}", call.function.name, call.function.arguments);
                // Execute the tool using the global registry function
                match execute_tool(&call.function.name, &call.function.arguments) {
                    Ok(result) => println!("  -> Result: {}", result),
                    Err(e) => println!("  -> Error: {}", e),
                }
                // You would typically send this result back to the LLM in a follow-up message
            }
        }
        CompletionKind::Message { content } => {
            println!("Message from LLM: {}", content);
        }
    }
    
    Ok(())
}
```

**How `#[merco_tool]` Works:**

1.  Takes your regular Rust functions.
2.  Generates necessary structs for argument parsing (`serde::Deserialize`).
3.  Automatically determines the JSON parameter schema based on function signatures.
4.  Handles serialization/deserialization of arguments and return values to/from JSON strings.
5.  Uses the `ctor` crate to register the tool's definition (`Tool`) and its execution logic (`ToolExecutor`) in a global registry when your program starts.

**Key Utilities:**

*   `#[merco_tool]`: The attribute macro to apply to your functions.
*   `get_tools_by_names(&[&str]) -> Vec<Tool>`: Retrieves specific tool definitions from the registry by name.
*   `get_all_tools() -> Vec<Tool>`: Retrieves all registered tool definitions.
*   `execute_tool(&str, &str) -> Result<String, String>`: Executes a registered tool by name using its JSON argument string.

Supported parameter types: integers (`i8`, `i16`, `i32`, `i64`, etc.), floats (`f32`, `f64`), strings (`String`, `&str`), booleans (`bool`), and basic `Vec<T>` of these types.

### 5. Manual Tool Setup (Legacy / Advanced)

For more complex scenarios, you can still manually define tools:

```rust
use merco_llmproxy::{
    LlmConfig, Provider, get_provider,
    traits::{ChatMessage, CompletionKind, CompletionRequest, JsonSchema, Tool,
        ToolCallFunction, ToolCallRequest, TokenUsage},
};
use serde::Deserialize;
use serde_json::json;

// 1. Define your tool implementation and argument struct
#[derive(Deserialize)]
struct SumArgs { a: i64, b: i64 }
fn sum_numbers(a: i64, b: i64) -> i64 { a + b }

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 2. Define the tool structure for the LLM
    let sum_tool = Tool {
        name: "sum_numbers".to_string(),
        description: "Calculates the sum of two integers.".to_string(),
        parameters: JsonSchema {
            schema_type: "object".to_string(),
            properties: Some({
                let mut props = serde_json::Map::new();
                props.insert("a".to_string(), json!({ "type": "integer" }));
                props.insert("b".to_string(), json!({ "type": "integer" }));
                props
            }),
            required: Some(vec!["a".to_string(), "b".to_string()]),
        },
    };

    // 3. Configure Provider (e.g., OpenRouter - requires env var)
    let api_key = std::env::var("OPENROUTER_API_KEY")?;
    let config = LlmConfig::new(Provider::OpenAI)
        .with_base_url("https://openrouter.ai/api/v1".to_string())
        .with_api_key(api_key);
    let provider = get_provider(config)?;

    // 4. Create Request with Tools
    let request = CompletionRequest {
        model: "mistralai/mistral-7b-instruct-v0.1".to_string(), // Specify model here
        messages: vec![
            ChatMessage {
                role: "user".to_string(),
                content: Some("What is the sum of 123 and 456?".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        temperature: Some(0.1),
        max_tokens: Some(150),
        tools: Some(vec![sum_tool]), // Provide the tool
    };

    // 5. Make Request and Handle Response
    match provider.completion(request).await {
        Ok(response) => {
            println!("Finish Reason: {:?}", response.finish_reason);
            match response.kind {
                CompletionKind::Message { content } => {
                    println!("Response Content:\n{}", content);
                }
                CompletionKind::ToolCall { tool_calls } => {
                    println!("Tool Calls Requested:");
                    for call in tool_calls {
                        println!("  ID: {}, Function: {}", call.id, call.function.name);
                        if call.function.name == "sum_numbers" {
                            match serde_json::from_str::<SumArgs>(&call.function.arguments) {
                                Ok(args) => {
                                    let result = sum_numbers(args.a, args.b);
                                    println!("  -> Simulated Result: {}", result);
                                    // Next step: Send result back in a new request
                                    // with role="tool", tool_call_id=call.id, content=result.to_string()
                                }
                                Err(e) => eprintln!("   -> Arg parse error: {}", e),
                            }
                        }
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Completion Error: {}", e);
        }
    }
    Ok(())
}
```

## Running Examples

The code in `src/main.rs` contains example usage similar to the snippets above. You can run it using:

```bash
# For OpenAI/OpenRouter tests:
export OPENROUTER_API_KEY="your-key-here"

# For Ollama tests, ensure Ollama server is running with the model:
# ollama pull qwen3:4b 

cargo run
```

The `examples/tool_example.rs` demonstrates the usage of the `merco_tool` macro:

```bash
cargo run --example tool_example
```

## Contributing

Contributions are welcome! Please feel free to open issues or pull requests.

## License

*(Choose a license, e.g., MIT or Apache-2.0)*
