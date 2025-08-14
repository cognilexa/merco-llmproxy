//!
//! OpenAI Provider Implementation
//! 
//! This module provides the `OpenAIProvider` struct and its implementation 
//! of the `LlmProvider` trait for interacting with OpenAI-compatible APIs
//! (including OpenAI itself and proxies like OpenRouter).

use crate::config::{LlmConfig, Provider, APP_SITE_NAME, APP_SITE_URL};
use crate::traits::{
    ChatMessage, CompletionKind, CompletionRequest, CompletionResponse, CompletionStream,
    CompletionStreamChunk, JsonSchema, LlmProvider, ProviderError, StreamContentDelta, Tool,
    ToolCallFunction, ToolCallFunctionStreamDelta, ToolCallRequest, ToolCallStreamDelta, TokenUsage,
};
use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::TryStreamExt; // Keep TryStreamExt for stream processing
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{self, json, Value as JsonValue};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use serde::de::Error as DeError;

/// Base URL for the official OpenAI API.
const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
/// Default request timeout in seconds.
const DEFAULT_TIMEOUT_SECS: u64 = 120;

// --- OpenAI Specific API Structures ---

#[derive(Serialize, Debug)]
struct OpenAITool {
    #[serde(rename = "type")]
    tool_type: String, // Always "function"
    function: OpenAIFunctionDef,
}

#[derive(Serialize, Debug)]
struct OpenAIFunctionDef {
    name: String,
    description: String,
    parameters: JsonSchema,
}

#[derive(Serialize, Debug)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<JsonValue>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)] // Allow unused fields from API response
struct OpenAIChatResponse {
    // model: String, // Often unused
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)] // Allow unused fields from API response
struct OpenAIChoice {
    // index: u32, // Often unused
    message: OpenAIMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)] // Allow unused fields from API response
struct OpenAIMessage {
    // role: String, // Often unused
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCall>>,
}

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)] // Allow unused fields from API response
struct OpenAIToolCall {
    id: String,
    // tool_type: String, // Often unused (assumed "function")
    function: OpenAIFunctionCall,
}

#[derive(Deserialize, Debug, Clone)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String, // JSON string arguments
}

#[derive(Deserialize, Debug, Clone, Copy)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

// --- Streaming Structures ---

#[derive(Deserialize, Debug)]
#[allow(dead_code)] // Allow unused fields from API response
struct OpenAIChatStreamResponse {
    // model: String, // Often unused
    choices: Vec<OpenAIStreamChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)] // Allow unused fields from API response
struct OpenAIStreamChoice {
    // index: u32, // Often unused
    delta: OpenAIStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)] // Allow unused fields from API response
struct OpenAIStreamDelta {
    // role: Option<String>, // Often unused
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIStreamToolCallDelta>>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)] // Allow unused fields from API response
struct OpenAIStreamToolCallDelta {
    index: usize,
    id: Option<String>,
    // tool_type: Option<String>, // Often unused
    function: Option<OpenAIStreamFunctionDelta>,
}

#[derive(Deserialize, Debug)]
struct OpenAIStreamFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

// For parsing OpenAI's specific error structure
#[derive(Deserialize, Debug)]
struct OpenAIErrorResponse {
    error: OpenAIErrorDetail,
}
#[derive(Deserialize, Debug)]
struct OpenAIErrorDetail {
    message: String,
    // code: Option<String>,
    // param: Option<String>,
    // error_type: Option<String>, // Renamed from type
}

// --- Provider Implementation ---

/// Provides interaction with OpenAI-compatible LLM APIs.
///
/// Supports standard chat completion and non-streaming tool calls.
/// Streaming tool calls are currently disabled due to parsing complexities.
#[derive(Debug, Clone)]
pub struct OpenAIProvider {
    config: LlmConfig,
    client: Client,
    api_key: String,
    base_url: String,
}

impl OpenAIProvider {
    /// Creates a new OpenAI provider instance from the given configuration.
    /// Panics if the configuration is missing the required API key or if the HTTP client fails to build.
    pub fn new(config: LlmConfig) -> Self {
        let api_key = config
            .api_key
            .clone()
            .expect("OpenAI provider requires an API key");

        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| OPENAI_BASE_URL.to_string());

        let client = Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .build()
            .expect("Failed to build Reqwest client");

        Self { config, client, api_key, base_url }
    }

    /// Builds the necessary HTTP headers for OpenAI API calls.
    /// Adds OpenRouter-specific headers if the base URL contains "openrouter".
    fn build_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .expect("Failed to create auth header"),
        );

        // Add OpenRouter-specific headers if using OpenRouter
        if self.base_url.to_lowercase().contains("openrouter") {
            headers.insert(
                "HTTP-Referer",
                HeaderValue::from_static(APP_SITE_URL),
            );
            headers.insert(
                "X-Title",
                HeaderValue::from_static(APP_SITE_NAME  ),
            );
        }

        headers
    }

    /// Maps the generic Tool structure to the OpenAI-specific format.
    fn map_tools_to_openai(tools: Option<&Vec<Tool>>) -> Option<Vec<OpenAITool>> {
        tools.map(|ts| {
            ts.iter()
                .map(|tool| OpenAITool {
                    tool_type: "function".to_string(), // Currently only support functions
                    function: OpenAIFunctionDef {
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        parameters: tool.parameters.clone(),
                    },
                })
                .collect()
        })
    }

    /// Maps the OpenAI usage structure to the generic TokenUsage structure.
    fn map_usage(usage: Option<OpenAIUsage>) -> Option<TokenUsage> {
         usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        })
    }

    /// Maps OpenAI tool calls to the generic ToolCallRequest structure.
    fn map_tool_calls(tool_calls: Vec<OpenAIToolCall>) -> Vec<ToolCallRequest> {
        tool_calls
            .into_iter()
            .map(|tc| ToolCallRequest {
                id: tc.id,
                tool_type: "function".to_string(),
                function: ToolCallFunction {
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                },
            })
            .collect()
    }

    /// Determines the final CompletionKind based on the message content, tool calls, and finish reason.
    fn determine_completion_kind(message: OpenAIMessage, finish_reason: Option<&str>) -> CompletionKind {
        match (message.content, message.tool_calls) {
            // If tool_calls are present, they take precedence, regardless of content.
            (_, Some(tool_calls)) => {
                CompletionKind::ToolCall { tool_calls: Self::map_tool_calls(tool_calls) }
            }
            // If no tool_calls but content is present, it's a message.
            (Some(content), None) => {
                CompletionKind::Message { content }
            }
            // If neither content nor tool_calls are present, determine based on finish reason.
            (None, None) => {
                match finish_reason {
                    Some("tool_calls") => {
                        // Model intended to call tools but didn't provide them (edge case?).
                        CompletionKind::ToolCall { tool_calls: vec![] }
                    }
                    _ => {
                        // Finished normally or other reason, but content was empty/null.
                        CompletionKind::Message { content: "".to_string() }
                    }
                }
            }
        }
    }
}

#[async_trait]
impl LlmProvider for OpenAIProvider {
    /// Generates a non-streaming completion, handling potential tool calls.
    async fn completion(&self, request: CompletionRequest) -> Result<CompletionResponse, ProviderError> {
        if self.config.provider != Provider::OpenAI {
            return Err(ProviderError::ConfigError(
                "Invalid provider configured for OpenAIProvider".to_string(),
            ));
        }

        let openai_request = OpenAIChatRequest {
            model: request.model.clone(),
            messages: request.messages.clone(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: false,
            tools: Self::map_tools_to_openai(request.tools.as_ref()),
            // Default to auto tool choice if tools are present, allows user override later
            tool_choice: request.tools.as_ref().map(|_| json!("auto")), 
        };

        let url = format!("{}/chat/completions", self.base_url);
        let headers = self.build_headers();

        let res = self.client.post(&url).headers(headers).json(&openai_request).send().await?;

        if !res.status().is_success() {
            let status = res.status().as_u16();
            let error_body = res.text().await.unwrap_or_else(|_| "Failed to read error body".to_string());
            // Try to parse OpenAI specific error
             let message = serde_json::from_str::<OpenAIErrorResponse>(&error_body)
                 .map(|e| e.error.message)
                 .unwrap_or(error_body); // Fallback to full body
            return Err(ProviderError::ApiError { status, message });
        }

        let openai_response: OpenAIChatResponse = res.json().await?;

        let first_choice = openai_response.choices.into_iter().next()
            .ok_or_else(|| ProviderError::ParseError(serde_json::Error::custom("No choices found in OpenAI response")))?;

        let usage = Self::map_usage(openai_response.usage);
        // Extract finish_reason before moving message into the helper
        let finish_reason = first_choice.finish_reason.clone(); 

        // Use the helper function to determine the kind (pass only message)
        let kind = Self::determine_completion_kind(first_choice.message, finish_reason.as_deref()); 

        Ok(CompletionResponse {
            kind,
            usage,
            finish_reason, // Use the extracted finish_reason
        })
    }

    /// Generates a streaming completion.
    /// NOTE: Tool calls are currently unsupported in streaming mode for this provider.
    async fn completion_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionStream, ProviderError> {
        // --- Temporarily disabled due to SSE parsing fragility ---
        if request.tools.is_some() {
            return Err(ProviderError::Unsupported(
                "Streaming tool calls are not currently supported by the OpenAI provider implementation.".to_string()
            ));
        }
        // --- End Temporary ---

        if self.config.provider != Provider::OpenAI {
            return Err(ProviderError::ConfigError(
                "Invalid provider configured for OpenAIProvider".to_string(),
            ));
        }

        let openai_request = OpenAIChatRequest {
            model: request.model.clone(),
            messages: request.messages.clone(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: true,
            tools: None, // Ensure tools are None for stream request
            tool_choice: None, // Ensure tool_choice is None for stream request
        };

        let url = format!("{}/chat/completions", self.base_url);
        let headers = self.build_headers();

        let res = self.client.post(&url).headers(headers).json(&openai_request).send().await?;

        if !res.status().is_success() {
            let status = res.status().as_u16();
            let error_body = res.text().await.unwrap_or_else(|_| "Failed to read error body".to_string());
             let message = serde_json::from_str::<OpenAIErrorResponse>(&error_body)
                 .map(|e| e.error.message)
                 .unwrap_or(error_body); 
            return Err(ProviderError::ApiError { status, message });
        }

        let sse_stream = res.bytes_stream().map_err(ProviderError::RequestError);

        // State for aggregating tool calls, wrapped for async stream handling
        let tool_call_aggregator = Arc::new(Mutex::new(HashMap::<usize, ToolCallStreamDelta>::new()));

        let chunk_stream = sse_stream.try_filter_map(move |chunk: Bytes| {
            let state_lock = Arc::clone(&tool_call_aggregator);
            async move {
                let lines = chunk.split(|&b| b == b'\n');
                let mut result_chunk: Option<CompletionStreamChunk> = None;
                let mut final_usage: Option<OpenAIUsage> = None;
                let mut final_reason: Option<String> = None;

                // Process each line in the chunk
                for line in lines {
                    if line.starts_with(b"data: ") {
                        let data = &line[6..];
                        if data.is_empty() || data == b"[DONE]" {
                            continue;
                        }

                        match serde_json::from_slice::<OpenAIChatStreamResponse>(data) {
                            Ok(openai_chunk) => {
                                if let Some(usage) = openai_chunk.usage {
                                    final_usage = Some(usage); // Capture final usage if present
                                }

                                if let Some(choice) = openai_chunk.choices.into_iter().next() {
                                     if let Some(reason) = choice.finish_reason {
                                         final_reason = Some(reason); // Capture final reason
                                     }

                                    // Lock mutex to process delta content
                                    let mut current_tool_calls = state_lock.lock().map_err(|_| {
                                        ProviderError::Unexpected("Mutex poisoned in stream processing".to_string())
                                    })?;

                                    if let Some(text_delta) = choice.delta.content {
                                        if !text_delta.is_empty() {
                                            result_chunk = Some(CompletionStreamChunk {
                                                delta: StreamContentDelta::Text(text_delta),
                                                usage: None,
                                                finish_reason: None,
                                            });
                                            current_tool_calls.clear(); // Clear tool state if text received
                                        }
                                    } else if let Some(tool_deltas) = choice.delta.tool_calls {
                                        let mut generic_deltas = Vec::new();
                                        for tool_delta in tool_deltas {
                                            let entry = current_tool_calls
                                                .entry(tool_delta.index)
                                                .or_insert_with(|| ToolCallStreamDelta {
                                                    index: tool_delta.index, id: None, function: None,
                                                });

                                            // Aggregate parts into the entry in the shared state
                                            if let Some(id) = tool_delta.id { entry.id = Some(id); }
                                            if let Some(func_delta) = tool_delta.function {
                                                let func_entry = entry.function.get_or_insert_with(|| {
                                                    ToolCallFunctionStreamDelta { name: None, arguments: None }
                                                });
                                                if let Some(name) = func_delta.name { func_entry.name = Some(name); }
                                                if let Some(args_chunk) = func_delta.arguments {
                                                     // DEBUG prints removed
                                                     let current_args = func_entry.arguments.get_or_insert_with(String::new);
                                                     current_args.push_str(&args_chunk);
                                                 }
                                            }
                                            // Add a *clone* of the current aggregated state to the output chunk
                                            generic_deltas.push(entry.clone()); 
                                        }
                                        // Only create a chunk if we actually processed deltas
                                        if !generic_deltas.is_empty() {
                                            result_chunk = Some(CompletionStreamChunk {
                                                delta: StreamContentDelta::ToolCallDelta(generic_deltas),
                                                usage: None,
                                                finish_reason: None,
                                            });
                                        }
                                    }
                                    // Mutex guard dropped here implicitly
                                }
                            }
                            Err(e) => {
                                eprintln!("Failed to parse OpenAI SSE chunk: {}, data: {}", e, String::from_utf8_lossy(data));
                                // Decide whether to stop stream on parse error
                                return Err(ProviderError::ParseError(e)); 
                            }
                        }
                    }
                }
                
                // If no data chunk was generated, but we got final usage/reason, create a final chunk
                if result_chunk.is_none() && (final_reason.is_some() || final_usage.is_some()) {
                     result_chunk = Some(CompletionStreamChunk {
                         delta: StreamContentDelta::Text("".to_string()), // Empty delta for final info
                         usage: Self::map_usage(final_usage),
                         finish_reason: final_reason,
                     });
                 }

                 Ok(result_chunk) // Return Option<CompletionStreamChunk>
            }
        });

        Ok(Box::pin(chunk_stream))
    }
} 