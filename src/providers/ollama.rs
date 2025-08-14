//!
//! Ollama Provider Implementation
//! 
//! Provides the `OllamaProvider` struct for interacting with local Ollama instances.
//! Supports non-streaming chat completions and non-streaming tool calls (via JSON mode).
//! Streaming tool calls are not supported as they require JSON mode, which Ollama disables for streaming.

use crate::config::{LlmConfig, Provider};
use crate::traits::{
    ChatMessage, ChatMessageRole, CompletionKind, CompletionRequest, CompletionResponse, CompletionStream, CompletionStreamChunk, LlmProvider, ProviderError, StreamContentDelta, TokenUsage, Tool, ToolCallFunction, ToolCallRequest
};
use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::TryStreamExt;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json;
use std::time::Duration;
use serde::de::Error as DeError;
use serde_json::Value as JsonValue;
use std::collections::HashMap;

/// Default base URL for a local Ollama instance.
const OLLAMA_DEFAULT_BASE_URL: &str = "http://localhost:11434";
/// Default request timeout in seconds.
const DEFAULT_TIMEOUT_SECS: u64 = 120;

// Internal structs mapping to Ollama's API
// We can reuse ChatMessage from traits.rs

#[derive(Serialize, Debug)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
}

#[derive(Serialize, Debug, Default)] // Default for easier optional creation
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>, 
}

// Non-streaming response
#[derive(Deserialize, Debug)]
struct OllamaChatResponse {
    model: String,
    created_at: String,
    message: ChatMessage, // Reuses the ChatMessage struct
    done: bool,
    // Timing/token info for non-streaming
    total_duration: Option<u64>,
    load_duration: Option<u64>,
    prompt_eval_count: Option<u32>,
    prompt_eval_duration: Option<u64>,
    eval_count: Option<u32>,      // Completion tokens
    eval_duration: Option<u64>,
}

// Streaming response chunk (newline-delimited JSON)
#[derive(Deserialize, Debug)]
struct OllamaChatStreamResponse {
    model: String,
    created_at: String,
    message: OllamaStreamMessage,
    done: bool,
    done_reason: Option<String>,

    // Timing/token info ONLY in the final chunk (when done = true)
    total_duration: Option<u64>,
    load_duration: Option<u64>,
    prompt_eval_count: Option<u32>,
    prompt_eval_duration: Option<u64>,
    eval_count: Option<u32>,      
    eval_duration: Option<u64>,
}

#[derive(Deserialize, Debug)]
struct OllamaStreamMessage {
    role: String,
    content: String, // This is the delta content for the stream
}

// Represents the *entire* JSON object returned when format=json
#[derive(Deserialize, Debug)]
struct OllamaJsonResponse {
    model: String,
    created_at: String,
    done: bool,
    total_duration: Option<u64>,
    load_duration: Option<u64>,
    prompt_eval_count: Option<u32>,
    prompt_eval_duration: Option<u64>,
    eval_count: Option<u32>,
    eval_duration: Option<u64>,
    message: Option<ChatMessage>,
    tool_calls: Option<Vec<OllamaToolCall>>,
}

// Standard non-streaming, non-json response
#[derive(Deserialize, Debug)]
struct OllamaStandardResponse {
    model: String,
    created_at: String,
    message: ChatMessage,
    done: bool,
    total_duration: Option<u64>,
    load_duration: Option<u64>,
    prompt_eval_count: Option<u32>,
    prompt_eval_duration: Option<u64>,
    eval_count: Option<u32>,
    eval_duration: Option<u64>,
}

// Define the structure we expect the model to put *inside* the message content
// Or potentially be the *entire* response in JSON mode
#[derive(Deserialize, Debug)]
struct OllamaToolCallPayload {
    tool_calls: Vec<OllamaToolCall>, 
}

#[derive(Deserialize, Debug)]
struct OllamaToolCall {
    id: String,
    function: OllamaToolFunction,
}

#[derive(Deserialize, Debug)]
struct OllamaToolFunction {
    name: String,
    arguments: JsonValue, // Expect arguments as a JSON Value (object), not pre-stringified
}

// --- Provider Implementation ---

/// Provides interaction with Ollama instances.
///
/// Supports non-streaming chat completion and non-streaming tool calls (using JSON mode).
#[derive(Debug, Clone)]
pub struct OllamaProvider {
    config: LlmConfig,
    client: Client,
    base_url: String,
}

impl OllamaProvider {
    /// Creates a new Ollama provider instance.
    pub fn new(config: LlmConfig) -> Self {
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| OLLAMA_DEFAULT_BASE_URL.to_string());

        let client = Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .build()
            .expect("Failed to build Reqwest client");

        // Note: Ollama doesn't typically use an API key, but config validation
        // might check for base_url presence.
        Self { config, client, base_url }
    }

    /// Builds standard HTTP headers for Ollama requests.
    fn build_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        // No Authorization header needed for default Ollama
        headers
    }

    /// Creates the Ollama options structure from the generic request.
    fn create_ollama_options(request: &CompletionRequest) -> Option<OllamaOptions> {
        let options = OllamaOptions {
            temperature: request.temperature,
            num_predict: request.max_tokens,
            // Map other generic options to Ollama options here
        };
        // Only return Some if at least one option is set
        if options.temperature.is_some() || options.num_predict.is_some() {
            Some(options)
        } else {
            None
        }
    }

    /// Formats tool definitions into a string suitable for inclusion in a system prompt.
    fn format_tools_for_prompt(tools: &[Tool]) -> String {
        let mut tool_desc = String::from("You have access to the following tools. Use them if necessary by outputting ONLY a JSON object with a single key 'tool_calls' containing a list of calls. Each call object in the list should have 'id' (a unique lowercase string), and 'function' containing 'name' (the tool name) and 'arguments' (a JSON object matching the tool's parameters schema). Do not output any other text, explanation, or markdown formatting around the JSON object.\n\nAvailable Tools:\n");
        for tool in tools {
            tool_desc.push_str(&format!("- Name: {}\n", tool.name));
            tool_desc.push_str(&format!("  Description: {}\n", tool.description));
            match serde_json::to_string_pretty(&tool.parameters) {
                Ok(params) => tool_desc.push_str(&format!("  Parameters Schema: {}\n", params)),
                Err(_) => tool_desc.push_str("  Parameters Schema: (Failed to format)\n"),
            }
        }
        tool_desc
    }

    /// Calculates token usage if prompt and completion counts are available.
    fn calculate_usage(prompt_tokens: Option<u32>, completion_tokens: Option<u32>) -> Option<TokenUsage> {
        match (prompt_tokens, completion_tokens) {
            (Some(pt), Some(ct)) => Some(TokenUsage {
                prompt_tokens: pt,
                completion_tokens: ct,
                total_tokens: pt + ct,
            }),
            _ => None,
        }
    }

    /// Maps Ollama-specific tool calls (parsed from JSON) to the generic ToolCallRequest structure.
    fn map_ollama_tool_calls(ollama_calls: Vec<OllamaToolCall>) -> Vec<ToolCallRequest> {
        ollama_calls.into_iter().map(|call| {
            ToolCallRequest {
                id: call.id,
                tool_type: "function".to_string(),
                function: ToolCallFunction {
                    name: call.function.name,
                    arguments: match call.function.arguments {
                        JsonValue::String(s) => s,
                        other => serde_json::to_string(&other).unwrap_or_default(),
                    }
                },
            }
        }).collect()
    }
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    /// Generates a non-streaming completion, potentially using JSON mode for tool calls.
    async fn completion(&self, request: CompletionRequest) -> Result<CompletionResponse, ProviderError> {
        if self.config.provider != Provider::Ollama && self.config.provider != Provider::Custom {
             return Err(ProviderError::ConfigError(
                 "Invalid provider configured for OllamaProvider".to_string(),
             ));
        }

        let mut original_messages = request.messages.clone();
        let mut use_json_format = false;

        // Modify prompt and set format if tools are present
        if let Some(tools) = &request.tools {
            if !tools.is_empty() {
                use_json_format = true;
                let tool_prompt = Self::format_tools_for_prompt(tools);

                // Find or create a system prompt in the original messages
                if let Some(system_message) = original_messages.iter_mut().find(|m| m.role == ChatMessageRole::System) {
                    let existing_content = system_message.content.take().unwrap_or_default();
                    system_message.content = Some(format!("{}\n\n{}", existing_content, tool_prompt));
                } else {
                    // Prepend a new system prompt
                    original_messages.insert(0, ChatMessage {
                        role: ChatMessageRole::System,
                        content: Some(tool_prompt),
                        tool_calls: None, // System prompts don't have tool calls
                        tool_call_id: None,
                    });
                }
            }
        }

        // Create a sanitized version of messages specifically for the Ollama API request,
        // removing fields Ollama doesn't expect in the input history.
        let messages_for_ollama_request: Vec<ChatMessage> = original_messages
            .into_iter()
            .map(|mut msg| {
                // Ollama API doesn't use tool_calls or tool_call_id in the request messages list.
                // Keep content and role.
                msg.tool_calls = None;
                // While Ollama doesn't use tool_call_id either, keeping it doesn't seem to cause errors
                // based on current Ollama API behavior, but we could clear it too if needed.
                // msg.tool_call_id = None;
                msg
            })
            .collect();


        let ollama_request = OllamaChatRequest {
            model: request.model.clone(),
            messages: messages_for_ollama_request, // Use the sanitized messages
            stream: false,
            format: if use_json_format { Some("json".to_string()) } else { None },
            options: Self::create_ollama_options(&request),
        };

        let url = format!("{}/api/chat", self.base_url);
        let headers = self.build_headers();

        let res = self
            .client
            .post(&url)
            .headers(headers)
            .json(&ollama_request)
            .send()
            .await?;

        if !res.status().is_success() {
            let status = res.status().as_u16();
            let error_body = res.text().await.unwrap_or_else(|_| "Failed to read error body".to_string());
            let message = serde_json::from_str::<HashMap<String, String>>(&error_body)
                .ok()
                .and_then(|json| json.get("error").cloned())
                .unwrap_or(error_body);
            return Err(ProviderError::ApiError { status, message });
        }

        // Handle response based on whether JSON format was requested
        if use_json_format {
            let raw_json_response: JsonValue = res.json().await?;

            // Try to parse the whole thing as our expected structure first
            match serde_json::from_value::<OllamaJsonResponse>(raw_json_response.clone()) {
                Ok(ollama_response) => {
                    let usage = Self::calculate_usage(ollama_response.prompt_eval_count, ollama_response.eval_count);

                    // Check primary tool_calls field first
                    if let Some(tool_calls) = ollama_response.tool_calls {
                        Ok(CompletionResponse {
                            kind: CompletionKind::ToolCall { tool_calls: Self::map_ollama_tool_calls(tool_calls) },
                            usage,
                            finish_reason: if ollama_response.done { Some("tool_calls".to_string()) } else { None },
                        })
                    } 
                    // If no top-level tool_calls, check if the *message content* contains it
                    else if let Some(message) = ollama_response.message {
                        if let Some(content_str) = &message.content {
                             // Attempt to parse the message content as JSON containing tool_calls
                             match serde_json::from_str::<OllamaToolCallPayload>(content_str) {
                                 Ok(tool_payload) => {
                                     // Ensure arguments are strings
                                      Ok(CompletionResponse {
                                         kind: CompletionKind::ToolCall { tool_calls: Self::map_ollama_tool_calls(tool_payload.tool_calls) },
                                         usage,
                                         finish_reason: if ollama_response.done { Some("tool_calls".to_string()) } else { None },
                                     })
                                 }
                                 Err(_) => {
                                     // Content wasn't the expected tool call JSON, treat as regular message
                                     Ok(CompletionResponse {
                                         kind: CompletionKind::Message { content: content_str.clone() },
                                         usage,
                                         finish_reason: if ollama_response.done { Some("stop".to_string()) } else { None },
                                     })
                                 }
                             }
                        } else {
                             // Message content was null, treat as empty message
                             Ok(CompletionResponse {
                                 kind: CompletionKind::Message { content: "".to_string() },
                                 usage,
                                 finish_reason: if ollama_response.done { Some("stop".to_string()) } else { None },
                             })
                        }
                    } else {
                         // JSON response didn't match expected structures
                         Err(ProviderError::ParseError(serde_json::Error::custom(
                             "Ollama JSON response did not contain expected 'message' or 'tool_calls' field."
                         )))
                    }
                }
                Err(_) => {
                    // Failed to parse as OllamaJsonResponse, maybe it's just the tool call payload directly?
                    match serde_json::from_value::<OllamaToolCallPayload>(raw_json_response) {
                        Ok(tool_payload) => {
                             // Estimate usage? Difficult without the standard response fields.
                             let usage = None; 
                              Ok(CompletionResponse {
                                 kind: CompletionKind::ToolCall { tool_calls: Self::map_ollama_tool_calls(tool_payload.tool_calls) },
                                 usage,
                                 finish_reason: Some("tool_calls".to_string()), // Assume tool call finish
                             })
                        }
                        Err(e) => {
                            // Couldn't parse as standard response or tool call payload
                             Err(ProviderError::ParseError(e))
                        }
                    }
                }
            }
        } else {
            // Standard non-JSON response parsing
            let ollama_response: OllamaStandardResponse = res.json().await?;
            let usage = Self::calculate_usage(ollama_response.prompt_eval_count, ollama_response.eval_count);
            Ok(CompletionResponse {
                kind: CompletionKind::Message { content: ollama_response.message.content.unwrap_or_default() },
                usage,
                finish_reason: if ollama_response.done { Some("stop".to_string()) } else { None },
            })
        }
    }

    /// Generates a streaming completion (tool calls unsupported).
    async fn completion_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionStream, ProviderError> {
        // Keep tool check for streaming because format=json disables it
        if request.tools.is_some() {
            return Err(ProviderError::Unsupported(
                "Streaming tool calls are not currently supported by the Ollama provider (requires format=json which disables streaming).".to_string()
            ));
        }

        if self.config.provider != Provider::Ollama && self.config.provider != Provider::Custom {
             return Err(ProviderError::ConfigError(
                 "Invalid provider configured for OllamaProvider".to_string(),
             ));
        }

        let ollama_request = OllamaChatRequest {
            model: request.model.clone(),
            messages: request.messages.clone(), // Use original messages for streaming
            stream: true,
            format: None, // Cannot use JSON format with streaming
            options: Self::create_ollama_options(&request),
        };

        let url = format!("{}/api/chat", self.base_url);
        let headers = self.build_headers();

        let res = self
            .client
            .post(&url)
            .headers(headers)
            .json(&ollama_request)
            .send()
            .await?;

        if !res.status().is_success() {
            let status = res.status().as_u16();
            let error_body = res.text().await.unwrap_or_else(|_| "Failed to read error body".to_string());
            let message = serde_json::from_str::<HashMap<String, String>>(&error_body)
                .ok()
                .and_then(|json| json.get("error").cloned())
                .unwrap_or(error_body);
            return Err(ProviderError::ApiError { status, message });
        }

        // Process the newline-delimited JSON stream
        let byte_stream = res.bytes_stream().map_err(ProviderError::RequestError);

        let chunk_stream = byte_stream.try_filter_map(|chunk: Bytes| async move {
            // Need to handle potential partial JSON objects across chunks if lines are split
            // For simplicity here, we assume each chunk contains whole lines
            // A more robust solution would buffer incomplete lines
            let lines = chunk.split(|&b| b == b'\n');
            let mut result_chunk: Option<CompletionStreamChunk> = None;

            for line in lines {
                if line.is_empty() { continue; }

                match serde_json::from_slice::<OllamaChatStreamResponse>(line) {
                    Ok(ollama_chunk) => {
                        let delta_content = ollama_chunk.message.content;
                        let usage = Self::calculate_usage(ollama_chunk.prompt_eval_count, ollama_chunk.eval_count);
                        let finish_reason = ollama_chunk.done_reason;

                         // Send a chunk if there's content or if it's the final chunk
                         if !delta_content.is_empty() || ollama_chunk.done {
                             result_chunk = Some(CompletionStreamChunk {
                                 delta: StreamContentDelta::Text(delta_content),
                                 usage,
                                 finish_reason,
                             });
                         }
                    }
                    Err(e) => {
                        // Log error and potentially yield an error
                        eprintln!(
                            "Failed to parse Ollama stream chunk: {:?}, data: {}",
                            e,
                            String::from_utf8_lossy(line)
                        );
                        // If it's a parsing error, maybe we should stop the stream
                         return Err(ProviderError::ParseError(e));
                    }
                }
            }
            Ok(result_chunk) // Return the processed chunk (if any) for this Bytes item
        });

        Ok(Box::pin(chunk_stream))
    }
} 