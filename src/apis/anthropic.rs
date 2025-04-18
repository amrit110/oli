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

// Anthropic API models
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum AnthropicContent {
    #[serde(rename = "text")]
    Text { text: String },

    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },

    #[serde(rename = "tool_result")]
    ToolResult {
        #[serde(rename = "tool_use_id")]
        tool_call_id: String,
        content: String,
    },
}

// The AnthropicToolUse struct is no longer needed as we're using AnthropicContent::ToolUse

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicTool {
    name: String,
    description: Option<String>,
    #[serde(rename = "input_schema")]
    schema: Value,
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
    system: Option<String>,
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
                                &format!(
                                    "Anthropic API rate limited or overloaded: {}",
                                    error_body
                                )
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
                            "Failed to send request to Anthropic after {} retries: {}",
                            retries, e
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
            HeaderValue::from_str(&format!("Bearer {}", api_key))?,
        );
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        headers.insert("x-api-key", HeaderValue::from_str(&api_key)?);

        let client = ReqwestClient::builder().default_headers(headers).build()?;

        // Default to Claude 3.7 Sonnet as the latest model with tooling capabilities
        let model = model.unwrap_or_else(|| "claude-3-7-sonnet-20250219".to_string());

        Ok(Self {
            client,
            model,
            api_base: "https://api.anthropic.com/v1/messages".to_string(),
        })
    }

    fn extract_system_message(&self, messages: &[Message]) -> Option<String> {
        messages
            .iter()
            .find(|msg| msg.role == "system")
            .map(|system_msg| system_msg.content.clone())
    }

    fn convert_messages(&self, messages: Vec<Message>) -> Vec<AnthropicMessage> {
        messages
            .into_iter()
            .filter(|msg| msg.role != "system") // Filter out system messages
            .map(|msg| AnthropicMessage {
                role: msg.role,
                content: vec![AnthropicContent::Text { text: msg.content }],
            })
            .collect()
    }

    fn convert_tool_definitions(
        &self,
        tools: Vec<crate::apis::api_client::ToolDefinition>,
    ) -> Vec<AnthropicTool> {
        tools
            .into_iter()
            .map(|tool| {
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

                AnthropicTool {
                    name: tool.name,
                    description: Some(tool.description),
                    schema: Value::Object(schema),
                }
            })
            .collect()
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
                "Anthropic API error: {} - {}",
                status, error_text
            ))
            .into());
        }

        // Get the response as a string first for debugging
        let response_text = response.text().await.map_err(|e| {
            let error_msg = format!("Failed to get response text: {}", e);
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
                let error_msg = format!("Failed to parse Anthropic response: {}", e);
                eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
                AppError::Other(error_msg)
            })?;

        // Extract content from response
        let mut text_content = String::new();

        // Look for text content in the response
        for content_item in &anthropic_response.content {
            if let AnthropicContent::Text { text } = content_item {
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
                    }],
                };

                // Create a tool result message (from user) with proper tool_result content
                let tool_result_msg = AnthropicMessage {
                    role: "user".to_string(),
                    content: vec![AnthropicContent::ToolResult {
                        tool_call_id: tool_call_id.clone(),
                        content: result.output.clone(),
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
                "Anthropic API error: {} - {}",
                status, error_text
            ))
            .into());
        }

        // Get the response as a string first for debugging
        let response_text = response.text().await.map_err(|e| {
            let error_msg = format!("Failed to get response text: {}", e);
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
                let error_msg = format!("Failed to parse Anthropic response: {}", e);
                eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
                AppError::Other(error_msg)
            })?;

        // First extract tool calls from content
        let mut tool_calls_vec = Vec::new();
        let mut text_content = String::new();

        // Process each content item
        for content_item in &anthropic_response.content {
            match content_item {
                AnthropicContent::Text { text } => {
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
