use merco_llmproxy::{
    ChatMessage, CompletionKind, CompletionRequest, LlmConfig, Provider, get_provider,
    merco_tool, get_all_tools, execute_tool,
};
use std::error::Error;

// Define a simple tool using the macro
#[merco_tool(description = "Adds two numbers together")]
fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}

// Define another tool
#[merco_tool(description = "Multiplies two numbers together")]
fn multiply_numbers(a: f64, b: f64) -> f64 {
    a * b
}

// Define a string-based tool
#[merco_tool(description = "Concatenates two strings")]
fn concat_strings(first: String, second: String) -> String {
    format!("{}{}", first, second)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // At this point, the tools are registered automatically
    
    // Get all registered tools
    let tools = get_all_tools();
    
    println!("Registered tools:");
    for tool in &tools {
        println!(" - {}: {}", tool.name, tool.description);
        println!("   Parameters: {:?}", tool.parameters);
    }
    
    // Execute a tool directly
    let add_result = execute_tool("add_numbers", r#"{"a": 5, "b": 7}"#)?;
    println!("\nDirect execution result of add_numbers(5, 7): {}", add_result);
    
    let multiply_result = execute_tool("multiply_numbers", r#"{"a": 3.5, "b": 2.0}"#)?;
    println!("Direct execution result of multiply_numbers(3.5, 2.0): {}", multiply_result);
    
    let concat_result = execute_tool("concat_strings", r#"{"first": "Hello, ", "second": "World!"}"#)?;
    println!("Direct execution result of concat_strings(\"Hello, \", \"World!\"): {}", concat_result);
    
    // Now, use these tools with an LLM (if available)
    if let Ok(api_key) = std::env::var("OPENROUTER_API_KEY") {
        println!("\nTesting LLM tool calling with OpenRouter:");
        
        // Create provider config
        let config = LlmConfig::new(
            Provider::OpenAI,
            "mistralai/mistral-7b-instruct-v0.1".to_string()
        )
        .with_base_url("https://openrouter.ai/api/v1".to_string())
        .with_api_key(api_key);
        
        let provider = get_provider(config)?;
        
        // Create a request with our tools
        let request = CompletionRequest {
            model: "mistralai/mistral-7b-instruct-v0.1".to_string(),
            messages: vec![
                ChatMessage {
                    role: "user".to_string(),
                    content: Some("What is 42 plus 17? Also, what is 8.5 multiplied by 3? Finally, can you concatenate 'Merco' and 'LLM'?".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
            temperature: Some(0.1),
            max_tokens: Some(300),
            tools: Some(tools), // Use our registered tools
        };
        
        // Make the request
        match provider.completion(request).await {
            Ok(response) => {
                println!("LLM Response:");
                match response.kind {
                    CompletionKind::Message { content } => {
                        println!("Text Response: {}", content);
                    }
                    CompletionKind::ToolCall { tool_calls } => {
                        println!("Tool Calls:");
                        for call in tool_calls {
                            println!("  Tool: {}", call.function.name);
                            println!("  Arguments: {}", call.function.arguments);
                            
                            // Execute the tool with the arguments from the LLM
                            match execute_tool(&call.function.name, &call.function.arguments) {
                                Ok(result) => println!("  Result: {}", result),
                                Err(e) => println!("  Execution Error: {}", e),
                            }
                        }
                    }
                }
            }
            Err(e) => {
                println!("LLM Request Error: {}", e);
            }
        }
    } else {
        println!("\nSkipping LLM test: OPENROUTER_API_KEY not set");
    }
    
    Ok(())
} 