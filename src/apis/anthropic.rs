use crate::apis::api_client::{ApiClient, CompletionOptions, Message, ToolCall, ToolResult};
use crate::app::logger::{format_log_with_color, LogLevel};
use crate::errors::AppError;
use anyhow::{Context, Result};
use async_trait::async_trait;
use rand;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::Client as ReqwestClient;
use reqwest::Response;
use serde::{Deserialize, Serialize};
use serde_json::{self, json, Value};
use std::env;
use std::time::Duration;

// Helper function to log usage information from Anthropic API
fn log_anthropic_usage(usage: &Value) {
    let mut input_tokens = usage
        .get("input_tokens")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let output_tokens = usage
        .get("output_tokens")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    // Include cache tokens in the total
    if let Some(cache_creation) = usage
        .get("cache_creation_input_tokens")
        .and_then(|v| v.as_i64())
    {
        input_tokens += cache_creation;
    }

    if let Some(cache_read) = usage
        .get("cache_read_input_tokens")
        .and_then(|v| v.as_i64())
    {
        input_tokens += cache_read;
    }

    eprintln!(
        "{}",
        format_log_with_color(
            LogLevel::Info,
            &format!(
                "Anthropic API usage: {} input tokens, {} output tokens, {} total tokens",
                input_tokens,
                output_tokens,
                input_tokens + output_tokens
            )
        )
    );
}

// Anthropic API models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
enum AnthropicContent {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    #[serde(rename = "tool_result")]
    ToolResult {
        #[serde(rename = "tool_use_id")]
        tool_call_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct CacheControl {
    #[serde(rename = "type")]
    cache_type: String,
}

// The AnthropicToolUse struct is no longer needed as we're using AnthropicContent::ToolUse

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct AnthropicTool {
    name: String,
    description: Option<String>,
    #[serde(rename = "input_schema")]
    schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    schema: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicToolChoice {
    #[serde(rename = "type")]
    choice_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<SystemContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<AnthropicResponseFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
enum SystemContent {
    String(String),
    Array(Vec<SystemBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct SystemBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicResponse {
    id: String,
    model: String,
    role: String,
    content: Vec<AnthropicContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    type_field: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequence: Option<String>,
}

pub struct AnthropicClient {
    client: ReqwestClient,
    model: String,
    api_base: String,
}

// Helper methods
impl AnthropicClient {
    /// Returns the model name being used by this client
    ///
    /// Primarily used for testing purposes.
    #[cfg(test)]
    pub(crate) fn get_model_name(&self) -> &str {
        &self.model
    }

    /// Creates an ephemeral cache control
    ///
    /// Helper function used for internal prompt caching
    fn create_ephemeral_cache() -> CacheControl {
        CacheControl {
            cache_type: "ephemeral".to_string(),
        }
    }
}

impl AnthropicClient {
    // Helper function to send a request with retry logic for overload errors
    async fn send_request_with_retry<T: serde::Serialize + Clone>(
        &self,
        request: &T,
    ) -> Result<Response> {
        // Implement retry logic with exponential backoff for 529 overload errors
        let mut retries = 0;
        let max_retries = 3; // Maximum number of retries
        let mut delay_ms = 1000; // Start with 1 second delay

        loop {
            let result = self.client.post(&self.api_base).json(request).send().await;

            match result {
                Ok(resp) => {
                    // If response is 429 (rate limit) or 529 (overloaded), retry
                    if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS
                        || resp.status().as_u16() == 529
                    {
                        if retries >= max_retries {
                            // Return the last error response if max retries reached
                            return Ok(resp);
                        }

                        // Extract retry-after header if available before cloning for the error body
                        let retry_after = resp
                            .headers()
                            .get("retry-after")
                            .and_then(|val| val.to_str().ok())
                            .and_then(|val| val.parse::<u64>().ok())
                            .unwrap_or(delay_ms);

                        // Clone the response for logging
                        let error_body = resp.text().await.unwrap_or_default();
                        eprintln!(
                            "{}",
                            format_log_with_color(
                                LogLevel::Warning,
                                &format!("Anthropic API rate limited or overloaded: {error_body}")
                            )
                        );

                        // Exponential backoff with jitter
                        let jitter = rand::random::<u64>() % 500;
                        let sleep_duration = Duration::from_millis(retry_after + jitter);

                        // Sleep and retry
                        tokio::time::sleep(sleep_duration).await;

                        // Increase delay for next retry
                        delay_ms = (delay_ms * 2).min(10000); // Cap at 10 seconds
                        retries += 1;
                        continue;
                    }

                    // For other status codes, return the response
                    return Ok(resp);
                }
                Err(e) => {
                    // For network errors, also use retry logic
                    if retries >= max_retries {
                        return Err(AppError::NetworkError(format!(
                            "Failed to send request to Anthropic after {retries} retries: {e}"
                        ))
                        .into());
                    }

                    // Exponential backoff with jitter
                    let jitter = rand::random::<u64>() % 500;
                    let sleep_duration = Duration::from_millis(delay_ms + jitter);
                    tokio::time::sleep(sleep_duration).await;

                    // Increase delay for next retry
                    delay_ms = (delay_ms * 2).min(10000); // Cap at 10 seconds
                    retries += 1;
                }
            }
        }
    }

    pub fn new(model: Option<String>) -> Result<Self> {
        // Try to get API key from environment
        let api_key = env::var("ANTHROPIC_API_KEY")
            .context("ANTHROPIC_API_KEY environment variable not set")?;

        Self::with_api_key(api_key, model)
    }

    pub fn with_api_key(api_key: String, model: Option<String>) -> Result<Self> {
        // Create new client with appropriate headers
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {api_key}"))?,
        );
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        headers.insert("x-api-key", HeaderValue::from_str(&api_key)?);

        let client = ReqwestClient::builder().default_headers(headers).build()?;

        // Default to Claude 3.7 Sonnet as the latest model with tooling capabilities
        let model = model.unwrap_or_else(|| "claude-sonnet-4-20250514".to_string());

        Ok(Self {
            client,
            model,
            api_base: "https://api.anthropic.com/v1/messages".to_string(),
        })
    }

    /// Extracts system message from the provided messages and formats it with cache control
    /// for prompt caching.
    ///
    /// This method finds the first message with the "system" role and formats it as a `SystemContent`
    /// with an ephemeral cache_control, allowing Claude to cache the system prompt.
    fn extract_system_message(&self, messages: &[Message]) -> Option<SystemContent> {
        messages
            .iter()
            .find(|msg| msg.role == "system")
            .map(|system_msg| {
                let system_block = SystemBlock {
                    block_type: "text".to_string(),
                    text: system_msg.content.clone(),
                    cache_control: Some(Self::create_ephemeral_cache()),
                };
                SystemContent::Array(vec![system_block])
            })
    }

    /// Converts internal message format to Anthropic's message format with cache control
    ///
    /// This method:
    /// 1. Filters out system messages (handled separately)
    /// 2. Formats each message as an AnthropicMessage
    /// 3. Adds cache_control to the last and second-to-last user messages for prompt caching
    fn convert_messages(&self, messages: Vec<Message>) -> Vec<AnthropicMessage> {
        let filtered_messages: Vec<Message> = messages
            .into_iter()
            .filter(|msg| msg.role != "system") // Filter out system messages
            .collect();

        let mut anthropic_messages = Vec::new();

        // Precompute the indices of user messages
        let user_indices: Vec<usize> = filtered_messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == "user")
            .map(|(i, _)| i)
            .collect();

        // Get the last and second-to-last user indices for caching optimization
        let last_user_index = user_indices.last().copied();
        let second_last_user_index = user_indices
            .get(user_indices.len().saturating_sub(2))
            .copied();

        // Use enumerated iterator to track position efficiently
        for (idx, msg) in filtered_messages.iter().enumerate() {
            let mut content = vec![AnthropicContent::Text {
                text: msg.content.clone(),
                cache_control: None,
            }];

            // Apply cache control to last and second-to-last user messages
            if let Some(last_idx) = last_user_index {
                // Always apply cache to the last user message
                if idx == last_idx {
                    content = vec![AnthropicContent::Text {
                        text: msg.content.clone(),
                        cache_control: Some(Self::create_ephemeral_cache()),
                    }];
                } else if let Some(second_last_idx) = second_last_user_index {
                    // Apply to second-to-last if it exists
                    if idx == second_last_idx {
                        content = vec![AnthropicContent::Text {
                            text: msg.content.clone(),
                            cache_control: Some(Self::create_ephemeral_cache()),
                        }];
                    }
                }
            }

            anthropic_messages.push(AnthropicMessage {
                role: msg.role.clone(),
                content,
            });
        }

        anthropic_messages
    }

    /// Converts internal tool definitions to Anthropic's format with cache control
    ///
    /// This method:
    /// 1. Converts each tool definition to Anthropic's format
    /// 2. Adds cache_control to the last tool definition for prompt caching
    /// 3. Creates a proper JSON Schema compliant schema for each tool
    fn convert_tool_definitions(
        &self,
        tools: Vec<crate::apis::api_client::ToolDefinition>,
    ) -> Vec<AnthropicTool> {
        let mut tool_specs = Vec::new();

        for (i, tool) in tools.iter().enumerate() {
            // Create a proper JSON Schema compliant schema object
            let mut schema = serde_json::Map::new();
            schema.insert(
                "$schema".to_string(),
                json!("https://json-schema.org/draft/2020-12/schema"),
            );
            schema.insert("type".to_string(), json!("object"));

            // Add properties and required fields if they exist in the original parameters
            if let Value::Object(params) = &tool.parameters {
                if let Some(props) = params.get("properties") {
                    schema.insert("properties".to_string(), props.clone());
                }

                if let Some(required) = params.get("required") {
                    schema.insert("required".to_string(), required.clone());
                }
            }

            // Add cache_control to the last tool spec
            let cache_control = if i == tools.len() - 1 {
                Some(Self::create_ephemeral_cache())
            } else {
                None
            };

            tool_specs.push(AnthropicTool {
                name: tool.name.clone(),
                description: Some(tool.description.clone()),
                schema: Value::Object(schema),
                cache_control,
            });
        }

        tool_specs
    }
}

#[async_trait]
impl ApiClient for AnthropicClient {
    async fn complete(&self, messages: Vec<Message>, options: CompletionOptions) -> Result<String> {
        // Extract system message if present
        let system_message = self.extract_system_message(&messages);
        let converted_messages = self.convert_messages(messages);

        let max_tokens = options.max_tokens.unwrap_or(2048) as usize;

        let mut request = AnthropicRequest {
            model: self.model.clone(),
            messages: converted_messages,
            max_tokens,
            system: system_message,
            temperature: options.temperature,
            top_p: options.top_p,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        // Add structured output format if specified in options
        if let Some(json_schema) = &options.json_schema {
            request.response_format = Some(AnthropicResponseFormat {
                format_type: "json".to_string(),
                schema: serde_json::from_str(json_schema).ok(),
            });
        }

        // Use our retry function instead of direct API call
        let response = self.send_request_with_retry(&request).await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AppError::NetworkError(format!(
                "Anthropic API error: {status} - {error_text}"
            ))
            .into());
        }

        // Get the response as a string first for debugging
        let response_text = response.text().await.map_err(|e| {
            let error_msg = format!("Failed to get response text: {e}");
            eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
            AppError::NetworkError(error_msg)
        })?;

        // Log response details
        eprintln!(
            "{}",
            format_log_with_color(
                LogLevel::Debug,
                &format!(
                    "Anthropic API response received: {} bytes",
                    response_text.len()
                )
            )
        );

        // Try to parse the response
        let anthropic_response: AnthropicResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                let error_msg = format!("Failed to parse Anthropic response: {e}");
                eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
                AppError::Other(error_msg)
            })?;

        // Extract content from response
        let mut text_content = String::new();

        // Look for text content in the response
        for content_item in &anthropic_response.content {
            if let AnthropicContent::Text { text, .. } = content_item {
                text_content = text.clone();
                break;
            }
        }

        // Return an error if no text content was found
        if text_content.is_empty() {
            let error_msg = "No text content in Anthropic response".to_string();
            eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
            return Err(AppError::LLMError(error_msg).into());
        }

        // Log usage information if available, including cache-related tokens
        if let Some(usage) = &anthropic_response.usage {
            log_anthropic_usage(usage);
        }

        let content = text_content;

        Ok(content)
    }

    async fn complete_with_tools(
        &self,
        messages: Vec<Message>,
        options: CompletionOptions,
        tool_results: Option<Vec<ToolResult>>,
    ) -> Result<(String, Option<Vec<ToolCall>>)> {
        // Extract system message if present
        let system_message = self.extract_system_message(&messages);
        let mut converted_messages = self.convert_messages(messages);

        // Add tool results if they exist
        if let Some(results) = tool_results {
            // For each tool result, we need to add corresponding messages
            for result in results {
                // Ensure we have a valid tool_call_id
                let tool_call_id = if result.tool_call_id.is_empty() {
                    // Generate a simple UUID-like string if no ID was provided
                    format!("tool-{}", rand::random::<u64>())
                } else {
                    result.tool_call_id.clone()
                };

                // Create a tool use message (from assistant)
                let tool_use_msg = AnthropicMessage {
                    role: "assistant".to_string(),
                    content: vec![AnthropicContent::ToolUse {
                        id: tool_call_id.clone(),
                        name: "tool".to_string(), // We don't have the original name
                        input: json!({}),         // We don't need the input for this
                        cache_control: None,
                    }],
                };

                // Create a tool result message (from user) with proper tool_result content
                let tool_result_msg = AnthropicMessage {
                    role: "user".to_string(),
                    content: vec![AnthropicContent::ToolResult {
                        tool_call_id: tool_call_id.clone(),
                        content: result.output.clone(),
                        cache_control: None,
                    }],
                };

                // Add both messages to the conversation
                converted_messages.push(tool_use_msg);
                converted_messages.push(tool_result_msg);
            }
        }

        let max_tokens = options.max_tokens.unwrap_or(2048) as usize;

        let mut request = AnthropicRequest {
            model: self.model.clone(),
            messages: converted_messages,
            max_tokens,
            system: system_message,
            temperature: options.temperature,
            top_p: options.top_p,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        // IMPORTANT: Add response_format only if json_schema exists AND tools don't exist
        // This fixes the "extra inputs are not permitted" error when using tools
        if let Some(json_schema) = &options.json_schema {
            // Only add response_format if we're not using tools
            if options.tools.is_none() {
                request.response_format = Some(AnthropicResponseFormat {
                    format_type: "json".to_string(),
                    schema: serde_json::from_str(json_schema).ok(),
                });
            }
        }

        // Add tools if they exist
        if let Some(tools) = options.tools {
            let converted_tools = self.convert_tool_definitions(tools);
            request.tools = Some(converted_tools);

            // Set tool choice based on option
            request.tool_choice = Some(AnthropicToolChoice {
                choice_type: if options.require_tool_use {
                    "required".to_string()
                } else {
                    "auto".to_string()
                },
            });
        }

        // Use our retry function instead of direct API call
        let response = self.send_request_with_retry(&request).await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AppError::NetworkError(format!(
                "Anthropic API error: {status} - {error_text}"
            ))
            .into());
        }

        // Get the response as a string first for debugging
        let response_text = response.text().await.map_err(|e| {
            let error_msg = format!("Failed to get response text: {e}");
            eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
            AppError::NetworkError(error_msg)
        })?;

        // Log response details
        eprintln!(
            "{}",
            format_log_with_color(
                LogLevel::Debug,
                &format!(
                    "Anthropic API response received: {} bytes",
                    response_text.len()
                )
            )
        );

        // Try to parse the response
        let anthropic_response: AnthropicResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                let error_msg = format!("Failed to parse Anthropic response: {e}");
                eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
                AppError::Other(error_msg)
            })?;

        // First extract tool calls from content
        let mut tool_calls_vec = Vec::new();
        let mut text_content = String::new();

        // Process each content item
        for content_item in &anthropic_response.content {
            match content_item {
                AnthropicContent::Text { text, .. } => {
                    // If we don't have a text content yet, use this one
                    if text_content.is_empty() {
                        text_content = text.clone();
                    }
                }
                AnthropicContent::ToolUse { name, input, .. } => {
                    // Add a tool call
                    tool_calls_vec.push(crate::apis::api_client::ToolCall {
                        id: None, // Anthropic doesn't provide IDs like OpenAI
                        name: name.clone(),
                        arguments: input.clone(),
                    });
                }
                AnthropicContent::ToolResult { .. } => {
                    // Tool results are not processed here, they're for the API to recognize tool result responses
                }
            }
        }

        // Log usage information if available, including cache-related tokens
        if let Some(usage) = &anthropic_response.usage {
            log_anthropic_usage(usage);
        }

        // If we didn't find any text content, use an empty string
        let content = if text_content.is_empty() {
            String::new()
        } else {
            text_content
        };

        // We no longer need to check a top-level tool_use field as all tool uses
        // will be in the content array already

        // Return None if no tool calls found, otherwise return the vector
        let tool_calls = if tool_calls_vec.is_empty() {
            None
        } else {
            Some(tool_calls_vec)
        };

        Ok((content, tool_calls))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apis::api_client::{Message, ToolDefinition};
    use serde_json::json;

    #[test]
    fn test_anthropic_model_name() {
        // Test that the default model name is correct when providing None
        // This doesn't make API calls, just tests the client setup logic
        let api_key = "test_api_key".to_string();
        let client = AnthropicClient::with_api_key(api_key, None).unwrap();

        // Verify the model name is the expected default
        assert_eq!(
            client.get_model_name(),
            "claude-sonnet-4-20250514",
            "Default model name should be claude-sonnet-4-20250514"
        );
    }

    #[test]
    fn test_anthropic_with_custom_model() {
        // Test that the custom model name is used correctly
        let api_key = "test_api_key".to_string();
        let model_name = "claude-sonnet-4-20250514".to_string();
        let client = AnthropicClient::with_api_key(api_key, Some(model_name.clone())).unwrap();

        // Verify the custom model name is used
        assert_eq!(
            client.get_model_name(),
            model_name,
            "Custom model name should be used"
        );
    }

    #[test]
    fn test_ephemeral_cache_creation() {
        // Test the helper method for creating cache control
        let cache = AnthropicClient::create_ephemeral_cache();

        assert_eq!(
            cache.cache_type, "ephemeral",
            "Cache type should be ephemeral"
        );
    }

    #[test]
    fn test_system_message_extraction() {
        // Create a test client
        let api_key = "test_api_key".to_string();
        let client = AnthropicClient::with_api_key(api_key, None).unwrap();

        // Create test messages including a system message
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
        ];

        // Extract the system message
        let system_content = client.extract_system_message(&messages);

        // Verify the system message was correctly extracted and formatted
        assert!(
            system_content.is_some(),
            "System message should be extracted"
        );

        if let Some(SystemContent::Array(blocks)) = system_content {
            assert_eq!(blocks.len(), 1, "Should contain exactly one system block");

            let block = &blocks[0];
            assert_eq!(block.block_type, "text", "Block type should be 'text'");
            assert_eq!(
                block.text, "You are a helpful assistant.",
                "Text content should match"
            );
            assert!(
                block.cache_control.is_some(),
                "Cache control should be present"
            );

            if let Some(cache) = &block.cache_control {
                assert_eq!(
                    cache.cache_type, "ephemeral",
                    "Cache type should be ephemeral"
                );
            }
        } else {
            panic!("System content should be an Array variant");
        }

        // Test with no system message
        let messages_without_system = vec![Message {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];

        let system_content = client.extract_system_message(&messages_without_system);
        assert!(
            system_content.is_none(),
            "No system message should be extracted"
        );
    }

    #[test]
    fn test_message_conversion_with_cache_control() {
        // Create a test client
        let api_key = "test_api_key".to_string();
        let client = AnthropicClient::with_api_key(api_key, None).unwrap();

        // Create test messages
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
            Message {
                role: "assistant".to_string(),
                content: "Hi there! How can I help you today?".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "Tell me about prompt caching".to_string(),
            },
        ];

        // Convert the messages
        let anthropic_messages = client.convert_messages(messages);

        // Verify messages are converted correctly
        assert_eq!(
            anthropic_messages.len(),
            3,
            "Should have 3 messages (system filtered out)"
        );

        // First user message should have cache control (second-to-last user)
        let first_user_msg = &anthropic_messages[0];
        assert_eq!(
            first_user_msg.role, "user",
            "First message should be a user message"
        );
        assert_eq!(
            first_user_msg.content.len(),
            1,
            "Should have one content block"
        );

        if let AnthropicContent::Text {
            text,
            cache_control,
        } = &first_user_msg.content[0]
        {
            assert_eq!(text, "Hello", "Text content should match");
            assert!(
                cache_control.is_some(),
                "First user message should have cache control"
            );
        } else {
            panic!("Content should be Text variant");
        }

        // Assistant message should not have cache control
        let assistant_msg = &anthropic_messages[1];
        assert_eq!(
            assistant_msg.role, "assistant",
            "Second message should be an assistant message"
        );

        if let AnthropicContent::Text {
            text,
            cache_control,
        } = &assistant_msg.content[0]
        {
            assert_eq!(
                text, "Hi there! How can I help you today?",
                "Text content should match"
            );
            assert!(
                cache_control.is_none(),
                "Assistant message should not have cache control"
            );
        } else {
            panic!("Content should be Text variant");
        }

        // Last user message should have cache control
        let last_user_msg = &anthropic_messages[2];
        assert_eq!(
            last_user_msg.role, "user",
            "Last message should be a user message"
        );

        if let AnthropicContent::Text {
            text,
            cache_control,
        } = &last_user_msg.content[0]
        {
            assert_eq!(
                text, "Tell me about prompt caching",
                "Text content should match"
            );
            assert!(
                cache_control.is_some(),
                "Last user message should have cache control"
            );
        } else {
            panic!("Content should be Text variant");
        }
    }

    #[test]
    fn test_message_conversion_edge_cases() {
        // Create a test client
        let api_key = "test_api_key".to_string();
        let client = AnthropicClient::with_api_key(api_key, None).unwrap();

        // Test with empty messages
        let empty_messages: Vec<Message> = vec![];
        let anthropic_messages = client.convert_messages(empty_messages);
        assert!(anthropic_messages.is_empty(), "Should produce no messages");

        // Test with only a system message (which will be filtered out)
        let only_system_message = vec![Message {
            role: "system".to_string(),
            content: "You are a helpful assistant.".to_string(),
        }];

        let anthropic_messages = client.convert_messages(only_system_message);
        assert!(anthropic_messages.is_empty(), "Should produce no messages");

        // Test with a single user message
        let single_user_message = vec![Message {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];

        let anthropic_messages = client.convert_messages(single_user_message);
        assert_eq!(anthropic_messages.len(), 1, "Should produce 1 message");

        // The single user message should have cache control as it's the last user message
        if let AnthropicContent::Text { cache_control, .. } = &anthropic_messages[0].content[0] {
            assert!(
                cache_control.is_some(),
                "Single user message should have cache control"
            );
        } else {
            panic!("Content should be Text variant");
        }
    }

    #[test]
    fn test_tool_definitions_conversion() {
        // Create a test client
        let api_key = "test_api_key".to_string();
        let client = AnthropicClient::with_api_key(api_key, None).unwrap();

        // Create test tools
        let tools = vec![
            ToolDefinition {
                name: "calculator".to_string(),
                description: "Calculate mathematical expressions".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate"
                        }
                    }
                }),
            },
            ToolDefinition {
                name: "weather".to_string(),
                description: "Get weather information".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get weather for"
                        }
                    }
                }),
            },
        ];

        // Convert the tools
        let anthropic_tools = client.convert_tool_definitions(tools);

        // Verify tools are converted correctly
        assert_eq!(anthropic_tools.len(), 2, "Should have 2 tools");

        // First tool should not have cache control
        let first_tool = &anthropic_tools[0];
        assert_eq!(
            first_tool.name, "calculator",
            "First tool should be the calculator"
        );
        assert_eq!(
            first_tool.description.as_ref().unwrap(),
            "Calculate mathematical expressions",
            "Description should match"
        );
        assert!(
            first_tool.cache_control.is_none(),
            "First tool should not have cache control"
        );

        // Schema should have required properties
        let schema = &first_tool.schema;
        assert!(
            schema.get("$schema").is_some(),
            "Schema should have $schema property"
        );
        assert!(
            schema.get("type").is_some(),
            "Schema should have type property"
        );
        assert!(
            schema.get("properties").is_some(),
            "Schema should have properties property"
        );

        // Last tool should have cache control
        let last_tool = &anthropic_tools[1];
        assert_eq!(
            last_tool.name, "weather",
            "Last tool should be the weather tool"
        );
        assert!(
            last_tool.cache_control.is_some(),
            "Last tool should have cache control"
        );

        if let Some(cache) = &last_tool.cache_control {
            assert_eq!(
                cache.cache_type, "ephemeral",
                "Cache type should be ephemeral"
            );
        }
    }

    #[test]
    fn test_tool_definitions_edge_cases() {
        // Create a test client
        let api_key = "test_api_key".to_string();
        let client = AnthropicClient::with_api_key(api_key, None).unwrap();

        // Test with empty tools
        let empty_tools: Vec<ToolDefinition> = vec![];
        let anthropic_tools = client.convert_tool_definitions(empty_tools);
        assert!(anthropic_tools.is_empty(), "Should produce no tools");

        // Test with a single tool
        let single_tool = vec![ToolDefinition {
            name: "calculator".to_string(),
            description: "Calculate mathematical expressions".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                }
            }),
        }];

        let anthropic_tools = client.convert_tool_definitions(single_tool);
        assert_eq!(anthropic_tools.len(), 1, "Should produce 1 tool");

        // The single tool should have cache control as it's the last tool
        assert!(
            anthropic_tools[0].cache_control.is_some(),
            "Single tool should have cache control"
        );

        // Test with tool that has no parameters properties
        let tool_without_properties = vec![ToolDefinition {
            name: "simple".to_string(),
            description: "Simple tool".to_string(),
            parameters: json!({
                "type": "object"
            }),
        }];

        let anthropic_tools = client.convert_tool_definitions(tool_without_properties);
        assert_eq!(anthropic_tools.len(), 1, "Should produce 1 tool");

        // Schema should still be valid
        let schema = &anthropic_tools[0].schema;
        assert!(
            schema.get("$schema").is_some(),
            "Schema should have $schema property"
        );
        assert!(
            schema.get("type").is_some(),
            "Schema should have type property"
        );
        assert!(
            schema.get("properties").is_none(),
            "Schema should not have properties property"
        );
    }

    #[test]
    fn test_caching_integration() {
        // This test simulates the complete flow to ensure all caching components work together
        let api_key = "test_api_key".to_string();
        let client = AnthropicClient::with_api_key(api_key, None).unwrap();

        // Create test messages and tools
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
            Message {
                role: "assistant".to_string(),
                content: "Hi there! How can I help you today?".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "Tell me about prompt caching".to_string(),
            },
        ];

        let tools = vec![ToolDefinition {
            name: "calculator".to_string(),
            description: "Calculate mathematical expressions".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                }
            }),
        }];

        // Extract system message
        let system_content = client.extract_system_message(&messages);
        assert!(
            system_content.is_some(),
            "System message should be extracted"
        );

        // Convert messages
        let anthropic_messages = client.convert_messages(messages.clone());
        assert_eq!(anthropic_messages.len(), 3, "Should have 3 messages");

        // Convert tools
        let anthropic_tools = client.convert_tool_definitions(tools);
        assert_eq!(anthropic_tools.len(), 1, "Should have 1 tool");

        // Verify cache control is added at each stage
        // System message
        if let Some(SystemContent::Array(blocks)) = &system_content {
            assert!(
                blocks[0].cache_control.is_some(),
                "System should have cache control"
            );
        }

        // Messages: first and last user message should have cache control
        let user_messages_with_cache = anthropic_messages
            .iter()
            .filter(|msg| msg.role == "user")
            .filter(|msg| {
                if let AnthropicContent::Text { cache_control, .. } = &msg.content[0] {
                    cache_control.is_some()
                } else {
                    false
                }
            })
            .count();

        assert_eq!(
            user_messages_with_cache, 2,
            "Two user messages should have cache control"
        );

        // Tool: the single tool should have cache control
        assert!(
            anthropic_tools[0].cache_control.is_some(),
            "Tool should have cache control"
        );
    }
}
