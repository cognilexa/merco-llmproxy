use async_trait::async_trait;
use futures::stream::Stream; // Requires the `futures` crate
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue; // For JSON Schema representation
use std::pin::Pin;
use thiserror::Error;

// --- Tool Calling Structures ---

/// Represents a tool (function) that the LLM can be instructed to call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// The name of the function to be called.
    pub name: String,
    /// A description of what the function does, used by the model to choose when and how to call it.
    pub description: String,
    /// The parameters the function accepts, described as a JSON Schema object.
    pub parameters: JsonSchema,
}

/// Represents a subset of JSON Schema for defining tool parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchema {
    /// The type of the schema (usually "object").
    #[serde(rename = "type")]
    pub schema_type: String,
    /// A map defining the properties (parameters) of the object.
    pub properties: Option<serde_json::Map<String, JsonValue>>,
    /// An array of strings listing the names of required properties.
    pub required: Option<Vec<String>>,
}

// --- Request/Response Structures ---

/// Represents a request to an LLM provider for chat completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// A list of messages comprising the conversation history.
    pub messages: Vec<ChatMessage>,
    /// The model identifier to use for completion.
    pub model: String,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Maximum number of tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// A list of tools the model may call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    // Consider adding tool_choice option later.
}

impl CompletionRequest {
    pub fn new(messages: Vec<ChatMessage>, model: String, temperature: Option<f32>, max_tokens: Option<u32>, tools: Option<Vec<Tool>>) -> Self {
        Self { messages, model, temperature, max_tokens, tools }
    }
}

/// Represents the role of a message sender in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChatMessageRole {
    /// A system message that provides instructions or context for the model.
    #[serde(rename = "system")]
    System,
    /// A user message that contains the actual input or query.
    #[serde(rename = "user")]
    User,
    /// An assistant message that contains the model's response.
    #[serde(rename = "assistant")]
    Assistant,
    /// A tool message that contains the result of a tool call.
    #[serde(rename = "tool")]
    Tool,
}

/// Represents a single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The role of the message sender (e.g., "system", "user", "assistant", "tool").
    pub role: ChatMessageRole,
    /// The text content of the message. Can be None for assistant messages requesting tool calls
    /// or for tool messages providing results.
    pub content: Option<String>,
    /// A list of tool calls requested by the assistant.
    /// Present only for `assistant` role messages when tools are called.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallRequest>>,
    /// The ID of the tool call this message is responding to.
    /// Present only for `tool` role messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn new(role: ChatMessageRole, content: Option<String>, tool_calls: Option<Vec<ToolCallRequest>>, tool_call_id: Option<String>) -> Self {
        Self { role, content, tool_calls, tool_call_id }
    }
    
    // Helper for creating a user message
    pub fn user(content: String) -> Self {
        Self { role: ChatMessageRole::User, content: Some(content), tool_calls: None, tool_call_id: None }
    }
    
    // Helper for creating a system message
    pub fn system(content: String) -> Self {
        Self { role: ChatMessageRole::System, content: Some(content), tool_calls: None, tool_call_id: None }
    }

    // Helper for creating an assistant message
    pub fn assistant(content: Option<String>, tool_calls: Option<Vec<ToolCallRequest>>) -> Self {
         Self { role: ChatMessageRole::Assistant, content, tool_calls, tool_call_id: None }
    }

    // Helper for creating a tool result message
    pub fn tool_result(tool_call_id: String, content: String) -> Self {
         Self { role: ChatMessageRole::Tool, content: Some(content), tool_calls: None, tool_call_id: Some(tool_call_id) }
    }
}

/// Represents a tool call requested by the LLM assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequest {
    /// A unique identifier for this specific tool call instance.
    pub id: String,
    /// The type of the tool call. Always "function" for now.
    #[serde(rename = "type")] // Ensure it serializes as "type"
    pub tool_type: String,
    /// The function details for the call.
    pub function: ToolCallFunction,
}

impl ToolCallRequest {
    pub fn new_function_call(id: String, function: ToolCallFunction) -> Self {
        Self {
            id,
            tool_type: "function".to_string(), // Always set to function
            function,
        }
    }
}

/// Details of the function being called in a `ToolCallRequest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    /// The name of the function to call.
    pub name: String,
    /// The arguments to call the function with, as a JSON string.
    pub arguments: String,
}

/// Represents the kind of result returned by a completion: either a message or tool calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CompletionKind {
    /// The LLM generated a text message.
    Message { content: String },
    /// The LLM requested one or more tool calls.
    ToolCall { tool_calls: Vec<ToolCallRequest> },
}

/// Represents the complete response from a non-streaming LLM completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// The kind of completion result (message or tool calls).
    #[serde(flatten)]
    pub kind: CompletionKind,
    /// Token usage information for the request (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    /// The reason the model stopped generating tokens (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Represents the kind of content delta in a streaming response chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamContentDelta {
    /// A chunk of text content.
    #[serde(rename = "content")]
    Text(String),
    /// Incremental information about tool calls being generated.
    #[serde(rename = "tool_calls")]
    ToolCallDelta(Vec<ToolCallStreamDelta>),
}

/// Represents incremental information about a single tool call within a stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallStreamDelta {
    /// The index of the tool call this delta belongs to (in case of multiple parallel calls).
    pub index: usize,
    /// The ID of the tool call (usually appears once).
    pub id: Option<String>,
    /// Incremental details of the function call.
    pub function: Option<ToolCallFunctionStreamDelta>,
}

/// Incremental details of the function being called in a `ToolCallStreamDelta`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunctionStreamDelta {
    /// The name of the function (usually appears once).
    pub name: Option<String>,
    /// A chunk of the JSON string arguments.
    pub arguments: Option<String>,
}

/// Represents a single chunk of data in a streaming LLM completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionStreamChunk {
    /// The content delta for this chunk (either text or tool call info).
    pub delta: StreamContentDelta,
    /// Token usage information (usually only present in the final chunk, if at all).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    /// The reason the model stopped (usually only present in the final chunk, if at all).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Represents token usage statistics for a completion request.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Tokens used in the prompt.
    pub prompt_tokens: u32,
    /// Tokens generated in the completion.
    pub completion_tokens: u32,
    /// Total tokens processed.
    pub total_tokens: u32,
}

/// Errors that can occur when interacting with LLM providers.
#[derive(Error, Debug)]
pub enum ProviderError {
    /// An error occurred during the underlying HTTP request.
    #[error("API request failed: {0}")]
    RequestError(#[from] reqwest::Error),
    /// The API returned an error response (e.g., 4xx, 5xx).
    #[error("API response error: {status}: {message}")]
    ApiError { status: u16, message: String },
    /// Failed to parse the JSON response from the API.
    #[error("Failed to parse API response: {0}")]
    ParseError(#[from] serde_json::Error),
    /// Configuration is invalid or missing required fields.
    #[error("Configuration error: {0}")]
    ConfigError(String),
    /// An error occurred during stream processing.
    #[error("Stream failed: {0}")]
    StreamError(String),
    /// A required configuration value was missing.
    #[error("Missing required configuration: {0}")]
    MissingConfig(String),
    /// Error related to the format or processing of tool use/calls.
    #[error("Tool use response format error: {0}")]
    ToolFormatError(String),
    /// The requested operation is not supported by the provider implementation.
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
    /// An unexpected internal error occurred.
    #[error("An unexpected error occurred: {0}")]
    Unexpected(String),
}

/// Type alias for the stream of completion chunks.
/// Uses dynamic dispatch (`dyn Stream`) and requires `Send` for async compatibility.
pub type CompletionStream =
    Pin<Box<dyn Stream<Item = Result<CompletionStreamChunk, ProviderError>> + Send>>;

/// The core asynchronous trait defining the interface for LLM providers.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Generates a non-streaming completion response.
    ///
    /// Takes a `CompletionRequest` and returns a `CompletionResponse`, which might contain
    /// either a text message or a request to call tools.
    async fn completion(&self, request: CompletionRequest) -> Result<CompletionResponse, ProviderError>;

    /// Generates a streaming completion response.
    ///
    /// Takes a `CompletionRequest` and returns a stream (`CompletionStream`) that yields
    /// `CompletionStreamChunk` results.
    async fn completion_stream(&self, request: CompletionRequest) -> Result<CompletionStream, ProviderError>;
} 