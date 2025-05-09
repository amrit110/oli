use crate::apis::api_client::{
    ApiClient, CompletionOptions, Message, ToolCall, ToolDefinition, ToolResult,
};
use crate::app::logger::{format_log_with_color, LogLevel};
use crate::errors::AppError;
use anyhow::Result;
use async_trait::async_trait;
use rand;

use reqwest::Client as ReqwestClient;
use serde::{Deserialize, Serialize};
use serde_json::{self, json, Value};
use std::time::Duration;

// Ollama API Types
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaMessage {
    role: String,
    #[serde(default)]
    #[serde(with = "content_string_or_object")]
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

// Custom serializer to handle content that might be a string or a complex object
mod content_string_or_object {
    use serde::{self, Deserialize, Deserializer, Serializer};
    use serde_json::Value;

    pub fn serialize<S>(content: &str, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(content)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<String, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(s) => Ok(s),
            _ => Ok(value.to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaToolCall {
    #[serde(default)]
    id: String,
    function: OllamaFunctionCall,
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    tool_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaFunctionCall {
    name: String,
    #[serde(with = "arguments_as_string_or_object")]
    arguments: String,
}

// Custom serde module for function arguments
mod arguments_as_string_or_object {
    use serde::{self, Deserialize, Deserializer, Serializer};
    use serde_json::Value;

    pub fn serialize<S>(arguments: &str, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(arguments)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<String, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;

        match value {
            Value::String(s) => Ok(s),
            _ => {
                // If it's not a string, convert the JSON to a string
                let json_str = serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string());
                Ok(json_str)
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OllamaFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OllamaTool>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaResponse {
    model: String,
    created_at: String,
    message: OllamaMessage,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_duration: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaListModelsResponse {
    models: Vec<OllamaModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModelInfo {
    pub name: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<OllamaModelDetails>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModelDetails {
    pub parameter_size: Option<String>,
    pub quantization_level: Option<String>,
    pub format: Option<String>,
    pub families: Option<Vec<String>>,
    pub description: Option<String>,
}

pub struct OllamaClient {
    client: ReqwestClient,
    model: String,
    api_base: String,
}

// Helper methods
impl OllamaClient {
    /// Returns the model name being used by this client
    ///
    /// Primarily used for testing purposes.
    #[cfg(test)]
    pub(crate) fn get_model_name(&self) -> &str {
        &self.model
    }

    /// Returns the API base URL used by this client
    ///
    /// Primarily used for testing purposes.
    #[cfg(test)]
    pub(crate) fn get_api_base(&self) -> &str {
        &self.api_base
    }
}

impl OllamaClient {
    pub fn new(model: Option<String>) -> Result<Self> {
        // Use the provided model name, with no default
        // If no model name provided, the client will still initialize
        // but the caller must specify a model when making a request
        let model_name = model.unwrap_or_default();

        // Check for OLLAMA_API_BASE environment variable first, fallback to localhost
        let api_base = std::env::var("OLLAMA_API_BASE")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());

        eprintln!("Initializing Ollama client at: {}", api_base);

        Self::with_base_url(model_name, api_base)
    }

    pub fn with_base_url(model: String, api_base: String) -> Result<Self> {
        // Build a simple client with only timeout configuration
        let client = ReqwestClient::builder()
            .timeout(Duration::from_secs(600)) // 10 minutes timeout for operations
            .build()
            .map_err(|e| {
                eprintln!("Failed to build reqwest client: {}", e);
                anyhow::anyhow!("Failed to build HTTP client: {}", e)
            })?;

        // Parse and normalize the API base URL
        let api_base = if api_base.starts_with("http://") || api_base.starts_with("https://") {
            api_base
        } else {
            format!("http://{}", api_base)
        };

        eprintln!("Using normalized Ollama API base URL: {}", api_base);

        Ok(Self {
            client,
            model,
            api_base,
        })
    }

    fn convert_messages(&self, messages: Vec<Message>) -> Vec<OllamaMessage> {
        messages
            .into_iter()
            .map(|msg| {
                // Convert standard messages to Ollama format
                OllamaMessage {
                    role: msg.role,
                    content: msg.content,
                    tool_calls: None,
                    tool_call_id: None,
                }
            })
            .collect()
    }

    fn convert_tool_definitions(&self, tools: Vec<ToolDefinition>) -> Vec<OllamaTool> {
        tools
            .into_iter()
            .map(|tool| OllamaTool {
                tool_type: "function".to_string(),
                function: OllamaFunction {
                    name: tool.name,
                    description: tool.description,
                    parameters: tool.parameters,
                },
            })
            .collect()
    }

    pub async fn list_models(&self) -> Result<Vec<OllamaModelInfo>> {
        let url = format!("{}/api/tags", self.api_base);

        eprintln!(
            "{}",
            format_log_with_color(
                LogLevel::Debug,
                &format!("Listing Ollama models from: {}", url)
            )
        );

        // More detailed debug information
        eprintln!(
            "{}",
            format_log_with_color(
                LogLevel::Debug,
                &format!("Using API base: {}, model: {}", self.api_base, self.model)
            )
        );

        // Print client information
        eprintln!(
            "{}",
            format_log_with_color(
                LogLevel::Debug,
                "Client configured with timeout: 300 seconds"
            )
        );

        // Try to send the request with better error handling
        let response = match self.client.get(&url).send().await {
            Ok(resp) => resp,
            Err(e) => {
                let error_msg = if e.is_connect() {
                    // Connection failed - likely Ollama is not running
                    format!("Failed to connect to Ollama server at {}. Make sure 'ollama serve' is running. Error: {}",
                            self.api_base, e)
                } else if e.is_timeout() {
                    format!("Request to Ollama timed out: {}", e)
                } else if e.is_request() {
                    format!("Failed to build request to Ollama: {}", e)
                } else {
                    format!("Failed to send request to Ollama: {}", e)
                };

                eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
                return Err(AppError::NetworkError(error_msg).into());
            }
        };

        // Check status code
        if !response.status().is_success() {
            let status = response.status();
            let error_text = match response.text().await {
                Ok(text) => text,
                Err(_) => "Failed to get error details".to_string(),
            };

            let error_msg = format!("Ollama API error: {} - {}", status, error_text);
            eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
            return Err(AppError::NetworkError(error_msg).into());
        }

        // Parse response text
        let response_text = match response.text().await {
            Ok(text) => text,
            Err(e) => {
                let error_msg = format!("Failed to get response text: {}", e);
                eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
                return Err(AppError::NetworkError(error_msg).into());
            }
        };

        eprintln!(
            "{}",
            format_log_with_color(
                LogLevel::Debug,
                &format!(
                    "Ollama API response received: {} bytes",
                    response_text.len()
                )
            )
        );

        // Try to parse the response
        match serde_json::from_str::<OllamaListModelsResponse>(&response_text) {
            Ok(models_response) => Ok(models_response.models),
            Err(e) => {
                let error_msg = format!(
                    "Failed to parse Ollama response: {}. Response text: {}",
                    e, response_text
                );
                eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
                Err(AppError::LLMError(error_msg).into())
            }
        }
    }
}

#[async_trait]
impl ApiClient for OllamaClient {
    async fn complete(&self, messages: Vec<Message>, options: CompletionOptions) -> Result<String> {
        let ollama_messages = self.convert_messages(messages);

        // Make sure we have a valid model name
        if self.model.is_empty() {
            return Err(anyhow::anyhow!("No model specified for Ollama request"));
        }
        let model_name = self.model.clone();

        let request = OllamaRequest {
            model: model_name.clone(),
            messages: ollama_messages,
            stream: false,
            temperature: options.temperature,
            top_p: options.top_p,
            options: None,
            format: if options.json_schema.is_some() {
                Some("json".to_string())
            } else {
                None
            },
            tools: None,
        };

        let url = format!("{}/api/chat", self.api_base);

        // Enhanced logging
        eprintln!(
            "{}",
            format_log_with_color(
                LogLevel::Debug,
                &format!(
                    "Sending request to Ollama API at {} with model: {}",
                    url, model_name
                )
            )
        );

        // Log request structure (sanitized to avoid logging entire messages)
        eprintln!(
            "{}",
            format_log_with_color(
                LogLevel::Debug,
                &format!(
                    "Request structure: model={}, messages={} items, stream=false",
                    model_name,
                    request.messages.len()
                )
            )
        );

        // Use match to provide more detailed error handling
        let response = match self.client.post(&url).json(&request).send().await {
            Ok(resp) => resp,
            Err(e) => {
                let error_msg = if e.is_connect() {
                    // Connection failed - likely Ollama is not running
                    format!("Failed to connect to Ollama server at {}. Make sure 'ollama serve' is running. Error: {}",
                        self.api_base, e)
                } else if e.is_timeout() {
                    format!("Request to Ollama timed out: {}", e)
                } else if e.is_request() {
                    format!("Failed to build request to Ollama: {}", e)
                } else if e.is_builder() {
                    format!("Failed to build HTTP request: {} - This may be due to a configuration issue with reqwest", e)
                } else {
                    format!("Failed to send request to Ollama: {}", e)
                };

                eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
                return Err(AppError::NetworkError(error_msg).into());
            }
        };

        // Handle non-success status codes
        if !response.status().is_success() {
            let status = response.status();

            // Get error details with better error handling
            let error_text = match response.text().await {
                Ok(text) => text,
                Err(_) => "Unknown error (failed to get error details)".to_string(),
            };

            let error_msg = format!("Ollama API error: {} - {}", status, error_text);
            eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
            return Err(AppError::NetworkError(error_msg).into());
        }

        // Get response text with better error handling
        let response_text = match response.text().await {
            Ok(text) => text,
            Err(e) => {
                let error_msg = format!("Failed to get response text: {}", e);
                eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
                return Err(AppError::NetworkError(error_msg).into());
            }
        };

        eprintln!(
            "{}",
            format_log_with_color(
                LogLevel::Debug,
                &format!(
                    "Ollama API response received: {} bytes",
                    response_text.len()
                )
            )
        );

        // Try to parse as a direct response with better fallback
        let ollama_response = match serde_json::from_str::<OllamaResponse>(&response_text) {
            Ok(resp) => {
                eprintln!(
                    "{}",
                    format_log_with_color(
                        LogLevel::Debug,
                        &format!(
                            "Successfully parsed standard Ollama response: model={}",
                            resp.model
                        )
                    )
                );
                resp
            }
            Err(e) => {
                // Log errors when parsing Ollama API response
                eprintln!(
                    "{}",
                    format_log_with_color(
                        LogLevel::Warning,
                        &format!("Failed to parse standard Ollama response: {}, attempting alternate parsing", e)
                    )
                );

                // Log the response text for debugging (truncated to avoid excessive logging)
                let preview = if response_text.len() > 100 {
                    format!("{}... [truncated]", &response_text[..100])
                } else {
                    response_text.clone()
                };

                eprintln!(
                    "{}",
                    format_log_with_color(
                        LogLevel::Debug,
                        &format!("Response text preview: {}", preview)
                    )
                );

                // Try to parse as a generic JSON value to extract what we need
                match serde_json::from_str::<serde_json::Value>(&response_text) {
                    Ok(value) => {
                        if let Some(message) = value.get("message") {
                            let role = message
                                .get("role")
                                .and_then(|r| r.as_str())
                                .unwrap_or("assistant")
                                .to_string();

                            // Extract content, which might be a string or object
                            let content = match message.get("content") {
                                Some(c) if c.is_string() => c.as_str().unwrap_or("").to_string(),
                                Some(c) => c.to_string(),
                                None => "".to_string(),
                            };

                            // Construct a valid OllamaResponse with the extracted data
                            OllamaResponse {
                                model: value
                                    .get("model")
                                    .and_then(|m| m.as_str())
                                    .unwrap_or("unknown")
                                    .to_string(),
                                created_at: value
                                    .get("created_at")
                                    .and_then(|t| t.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                message: OllamaMessage {
                                    role,
                                    content,
                                    tool_calls: None,
                                    tool_call_id: None,
                                },
                                done: value.get("done").and_then(|d| d.as_bool()).unwrap_or(true),
                                total_duration: None,
                                load_duration: None,
                                prompt_eval_duration: None,
                                eval_count: None,
                                eval_duration: None,
                            }
                        } else {
                            // If we didn't find a message, create a synthetic error response
                            eprintln!(
                                "{}",
                                format_log_with_color(
                                    LogLevel::Error,
                                    "Could not find 'message' field in Ollama response"
                                )
                            );

                            return Err(AppError::Other(format!(
                                "Failed to parse Ollama response: missing 'message' field. Response: {}",
                                preview
                            )).into());
                        }
                    }
                    Err(json_err) => {
                        // If we can't parse as JSON at all, return a clear error
                        return Err(AppError::Other(format!(
                            "Failed to parse Ollama response as JSON: {}. Raw response: {}",
                            json_err, preview
                        ))
                        .into());
                    }
                }
            }
        };

        Ok(ollama_response.message.content)
    }

    async fn complete_with_tools(
        &self,
        messages: Vec<Message>,
        options: CompletionOptions,
        tool_results: Option<Vec<ToolResult>>,
    ) -> Result<(String, Option<Vec<ToolCall>>)> {
        // Ensure we have a valid model
        if self.model.is_empty() {
            return Err(anyhow::anyhow!(
                "Model name is empty. Please select a valid Ollama model."
            ));
        }
        let model_name = self.model.clone();

        // Convert messages to Ollama format
        let mut ollama_messages = self.convert_messages(messages);

        // Add tool results if provided
        if let Some(results) = tool_results {
            for result in results {
                ollama_messages.push(OllamaMessage {
                    role: "tool".to_string(),
                    content: result.output,
                    tool_calls: None,
                    tool_call_id: Some(result.tool_call_id),
                });
            }
        }

        // Create the request payload
        let mut request = OllamaRequest {
            model: model_name.clone(),
            messages: ollama_messages,
            stream: false,
            temperature: options.temperature,
            top_p: options.top_p,
            options: None,
            format: if options.json_schema.is_some() {
                Some("json".to_string())
            } else {
                None
            },
            tools: None,
        };

        // Add tools if provided
        if let Some(tools) = options.tools {
            let converted_tools = self.convert_tool_definitions(tools);
            request.tools = Some(converted_tools);
        }

        let url = format!("{}/api/chat", self.api_base);

        // Enhanced logging
        eprintln!(
            "{}",
            format_log_with_color(
                LogLevel::Debug,
                &format!(
                    "Sending tool request to Ollama API at {} with model: {}",
                    url, model_name
                )
            )
        );

        // Log request structure (sanitized to avoid logging entire messages)
        eprintln!(
            "{}",
            format_log_with_color(
                LogLevel::Debug,
                &format!(
                    "Tool request structure: model={}, messages={} items, tools={} defined, stream=false",
                    model_name,
                    request.messages.len(),
                    request.tools.as_ref().map_or(0, |t| t.len())
                )
            )
        );

        // Use match to provide more detailed error handling
        let response = match self.client.post(&url).json(&request).send().await {
            Ok(resp) => resp,
            Err(e) => {
                let error_msg = if e.is_connect() {
                    // Connection failed - likely Ollama is not running
                    format!("Failed to connect to Ollama server at {}. Make sure 'ollama serve' is running. Error: {}",
                        self.api_base, e)
                } else if e.is_timeout() {
                    format!("Request to Ollama timed out: {}", e)
                } else if e.is_request() {
                    format!("Failed to build request to Ollama: {}", e)
                } else if e.is_builder() {
                    format!("Failed to build HTTP request: {} - This may be due to a configuration issue with reqwest", e)
                } else {
                    format!("Failed to send request to Ollama: {}", e)
                };

                eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
                return Err(AppError::NetworkError(error_msg).into());
            }
        };

        // Handle non-success status codes
        if !response.status().is_success() {
            let status = response.status();

            // Get error details with better error handling
            let error_text = match response.text().await {
                Ok(text) => text,
                Err(_) => "Unknown error (failed to get error details)".to_string(),
            };

            let error_msg = format!("Ollama API error: {} - {}", status, error_text);
            eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
            return Err(AppError::NetworkError(error_msg).into());
        }

        // Get response text with better error handling
        let response_text = match response.text().await {
            Ok(text) => text,
            Err(e) => {
                let error_msg = format!("Failed to get response text: {}", e);
                eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
                return Err(AppError::NetworkError(error_msg).into());
            }
        };

        eprintln!(
            "{}",
            format_log_with_color(
                LogLevel::Debug,
                &format!(
                    "Ollama API tool response received: {} bytes",
                    response_text.len()
                )
            )
        );

        // Try to parse as a direct response with better fallback
        let ollama_response = match serde_json::from_str::<OllamaResponse>(&response_text) {
            Ok(resp) => {
                eprintln!(
                    "{}",
                    format_log_with_color(
                        LogLevel::Debug,
                        &format!(
                            "Successfully parsed standard Ollama tool response: model={}",
                            resp.model
                        )
                    )
                );
                resp
            }
            Err(e) => {
                // Log errors when parsing Ollama API response
                eprintln!(
                    "{}",
                    format_log_with_color(
                        LogLevel::Warning,
                        &format!("Failed to parse standard Ollama tool response: {}, attempting alternate parsing", e)
                    )
                );

                // Log the response text for debugging (truncated to avoid excessive logging)
                let preview = if response_text.len() > 100 {
                    format!("{}... [truncated]", &response_text[..100])
                } else {
                    response_text.clone()
                };

                eprintln!(
                    "{}",
                    format_log_with_color(
                        LogLevel::Debug,
                        &format!("Tool response text preview: {}", preview)
                    )
                );

                // Try to parse as a generic JSON value to extract what we need
                match serde_json::from_str::<serde_json::Value>(&response_text) {
                    Ok(value) => {
                        if let Some(message) = value.get("message") {
                            let role = message
                                .get("role")
                                .and_then(|r| r.as_str())
                                .unwrap_or("assistant")
                                .to_string();

                            // Extract content, which might be a string or object
                            let content = match message.get("content") {
                                Some(c) if c.is_string() => c.as_str().unwrap_or("").to_string(),
                                Some(c) => c.to_string(),
                                None => "".to_string(),
                            };

                            // Construct a valid OllamaResponse with the extracted data
                            OllamaResponse {
                                model: value
                                    .get("model")
                                    .and_then(|m| m.as_str())
                                    .unwrap_or("unknown")
                                    .to_string(),
                                created_at: value
                                    .get("created_at")
                                    .and_then(|t| t.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                message: OllamaMessage {
                                    role,
                                    content,
                                    tool_calls: None,
                                    tool_call_id: None,
                                },
                                done: value.get("done").and_then(|d| d.as_bool()).unwrap_or(true),
                                total_duration: None,
                                load_duration: None,
                                prompt_eval_duration: None,
                                eval_count: None,
                                eval_duration: None,
                            }
                        } else {
                            // If we didn't find a message, create a synthetic error response
                            eprintln!(
                                "{}",
                                format_log_with_color(
                                    LogLevel::Error,
                                    "Could not find 'message' field in Ollama tool response"
                                )
                            );

                            return Err(AppError::Other(format!(
                                "Failed to parse Ollama tool response: missing 'message' field. Response: {}",
                                preview
                            )).into());
                        }
                    }
                    Err(json_err) => {
                        // If we can't parse as JSON at all, return a clear error
                        return Err(AppError::Other(format!(
                            "Failed to parse Ollama tool response as JSON: {}. Raw response: {}",
                            json_err, preview
                        ))
                        .into());
                    }
                }
            }
        };

        // Extract the content and tool calls from the response
        let content = ollama_response.message.content.clone();

        // Check for tool calls in the response
        if let Some(ollama_tool_calls) = ollama_response.message.tool_calls {
            if !ollama_tool_calls.is_empty() {
                eprintln!(
                    "{}",
                    format_log_with_color(
                        LogLevel::Debug,
                        &format!(
                            "Found {} tool calls in Ollama response",
                            ollama_tool_calls.len()
                        )
                    )
                );

                let tool_calls = ollama_tool_calls
                    .iter()
                    .map(|call| {
                        // Parse arguments as JSON
                        let arguments_result =
                            serde_json::from_str::<Value>(&call.function.arguments);
                        let arguments = match arguments_result {
                            Ok(args) => args,
                            Err(e) => {
                                eprintln!(
                                    "{}",
                                    format_log_with_color(
                                        LogLevel::Warning,
                                        &format!("Failed to parse tool arguments as JSON: {}. Using empty object instead.", e)
                                    )
                                );
                                json!({})
                            },
                        };

                        // Generate a random ID if one wasn't provided
                        let id = if call.id.is_empty() {
                            format!("ollama-tool-{}", rand::random::<u64>())
                        } else {
                            call.id.clone()
                        };

                        // Create a tool call
                        ToolCall {
                            id: Some(id),
                            name: call.function.name.clone(),
                            arguments,
                        }
                    })
                    .collect::<Vec<_>>();

                return Ok((String::new(), Some(tool_calls)));
            }
        }

        // Also try to check if the content itself contains a tool call in JSON format
        // This handles cases where Ollama doesn't properly format its tool_calls field
        // but still returns JSON in the content field that looks like a tool call
        let content_str = content.trim();
        if content_str.starts_with('{') && content_str.ends_with('}') {
            eprintln!(
                "{}",
                format_log_with_color(
                    LogLevel::Debug,
                    "Content appears to be JSON, checking for tool calls"
                )
            );

            if let Ok(json_value) = serde_json::from_str::<Value>(content_str) {
                // Check for OpenAI style tool calls
                if let Some(tool_calls) = json_value.get("tool_calls").and_then(|tc| tc.as_array())
                {
                    if !tool_calls.is_empty() {
                        eprintln!(
                            "{}",
                            format_log_with_color(
                                LogLevel::Debug,
                                &format!(
                                    "Found {} OpenAI-style tool calls in JSON content",
                                    tool_calls.len()
                                )
                            )
                        );

                        let calls = tool_calls
                            .iter()
                            .filter_map(|call| {
                                let id = call.get("id").and_then(|id| id.as_str()).unwrap_or("");
                                let function = call.get("function")?;
                                let name = function.get("name")?.as_str()?;
                                let arguments = function.get("arguments")?;

                                let args_str = arguments.as_str().unwrap_or("{}");
                                let args: Value =
                                    serde_json::from_str(args_str).unwrap_or(json!({}));

                                Some(ToolCall {
                                    id: Some(id.to_string()),
                                    name: name.to_string(),
                                    arguments: args,
                                })
                            })
                            .collect::<Vec<_>>();

                        if !calls.is_empty() {
                            return Ok((String::new(), Some(calls)));
                        }
                    }
                }

                // Check for the simpler/custom format that our old implementation expected
                if let (Some(tool_name), Some(tool_args)) = (
                    json_value.get("tool").and_then(|t| t.as_str()),
                    json_value.get("args"),
                ) {
                    eprintln!(
                        "{}",
                        format_log_with_color(
                            LogLevel::Debug,
                            &format!("Found simple tool call format with tool: {}", tool_name)
                        )
                    );

                    let tool_call = ToolCall {
                        id: Some(format!("ollama-tool-{}", rand::random::<u64>())),
                        name: tool_name.to_string(),
                        arguments: tool_args.clone(),
                    };

                    return Ok((String::new(), Some(vec![tool_call])));
                }
            }
        }

        // If no tool calls were found, just return the content
        eprintln!(
            "{}",
            format_log_with_color(
                LogLevel::Debug,
                "No tool calls found in response, returning content"
            )
        );

        Ok((content, None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apis::api_client::{Message, ToolDefinition};
    use serde_json::json;

    #[test]
    fn test_ollama_model_name() {
        // Test that the model name is set correctly
        let model_name = "llama3".to_string();
        let client = OllamaClient::new(Some(model_name.clone())).unwrap();

        // Verify the model name is used
        assert_eq!(
            client.get_model_name(),
            model_name,
            "Model name should match what was provided"
        );
    }

    #[test]
    fn test_ollama_empty_model_name() {
        // Test with empty model name
        let client = OllamaClient::new(None).unwrap();

        // Verify the model name is empty as expected
        assert_eq!(
            client.get_model_name(),
            "",
            "Model name should be empty when None is provided"
        );
    }

    #[test]
    fn test_ollama_custom_api_base() {
        // Test that the custom API base URL is used correctly
        let base_url = "http://custom-ollama-server:11434".to_string();
        let model_name = "mistral".to_string();
        let client = OllamaClient::with_base_url(model_name.clone(), base_url.clone()).unwrap();

        // Verify both model name and API base are set correctly
        assert_eq!(
            client.get_model_name(),
            model_name,
            "Model name should match what was provided"
        );
        assert_eq!(
            client.get_api_base(),
            base_url,
            "API base URL should match what was provided"
        );
    }

    #[test]
    fn test_ollama_api_base_normalization() {
        // Test that API base URLs without protocol get normalized
        let base_url = "localhost:11434".to_string();
        let model_name = "phi3".to_string();
        let client = OllamaClient::with_base_url(model_name, base_url).unwrap();

        // Verify API base URL gets normalized with http:// prefix
        assert_eq!(
            client.get_api_base(),
            "http://localhost:11434",
            "API base URL should be normalized with http:// prefix"
        );
    }

    #[test]
    fn test_convert_messages() {
        // Set up a client for testing conversion methods
        let client = OllamaClient::new(Some("llama3".to_string())).unwrap();

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
        ];

        // Convert the messages
        let ollama_messages = client.convert_messages(messages);

        // Verify messages are converted correctly
        assert_eq!(ollama_messages.len(), 3, "Should have 3 messages");

        // Check system message
        let system_msg = &ollama_messages[0];
        assert_eq!(
            system_msg.role, "system",
            "First message should be a system message"
        );
        assert_eq!(
            system_msg.content, "You are a helpful assistant.",
            "Content should match"
        );
        assert!(
            system_msg.tool_calls.is_none(),
            "System message should not have tool calls"
        );

        // Check user message
        let user_msg = &ollama_messages[1];
        assert_eq!(
            user_msg.role, "user",
            "Second message should be a user message"
        );
        assert_eq!(user_msg.content, "Hello", "Content should match");
        assert!(
            user_msg.tool_calls.is_none(),
            "User message should not have tool calls"
        );

        // Check assistant message
        let assistant_msg = &ollama_messages[2];
        assert_eq!(
            assistant_msg.role, "assistant",
            "Third message should be an assistant message"
        );
        assert_eq!(
            assistant_msg.content, "Hi there! How can I help you today?",
            "Content should match"
        );
        assert!(
            assistant_msg.tool_calls.is_none(),
            "Assistant message should not have tool calls"
        );
    }

    #[test]
    fn test_message_conversion_edge_cases() {
        // Set up a client for testing conversion methods
        let client = OllamaClient::new(Some("llama3".to_string())).unwrap();

        // Test with empty messages
        let empty_messages: Vec<Message> = vec![];
        let ollama_messages = client.convert_messages(empty_messages);
        assert!(ollama_messages.is_empty(), "Should produce no messages");

        // Test with a single message
        let single_message = vec![Message {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];

        let ollama_messages = client.convert_messages(single_message);
        assert_eq!(ollama_messages.len(), 1, "Should produce 1 message");
        assert_eq!(ollama_messages[0].role, "user", "Should be a user message");
        assert_eq!(ollama_messages[0].content, "Hello", "Content should match");
    }

    #[test]
    fn test_tool_definitions_conversion() {
        // Set up a client for testing conversion methods
        let client = OllamaClient::new(Some("llama3".to_string())).unwrap();

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
        let ollama_tools = client.convert_tool_definitions(tools);

        // Verify tools are converted correctly
        assert_eq!(ollama_tools.len(), 2, "Should have 2 tools");

        // Check first tool
        let calculator_tool = &ollama_tools[0];
        assert_eq!(
            calculator_tool.tool_type, "function",
            "Tool type should be function"
        );
        assert_eq!(
            calculator_tool.function.name, "calculator",
            "Name should match"
        );
        assert_eq!(
            calculator_tool.function.description, "Calculate mathematical expressions",
            "Description should match"
        );

        // Check function parameters
        let calculator_params = &calculator_tool.function.parameters;
        assert!(
            calculator_params.is_object(),
            "Parameters should be an object"
        );
        assert!(
            calculator_params.get("properties").is_some(),
            "Parameters should have properties"
        );
        assert!(
            calculator_params
                .get("properties")
                .unwrap()
                .get("expression")
                .is_some(),
            "Expression property should exist"
        );

        // Check second tool
        let weather_tool = &ollama_tools[1];
        assert_eq!(
            weather_tool.tool_type, "function",
            "Tool type should be function"
        );
        assert_eq!(weather_tool.function.name, "weather", "Name should match");
        assert_eq!(
            weather_tool.function.description, "Get weather information",
            "Description should match"
        );

        // Check function parameters
        let weather_params = &weather_tool.function.parameters;
        assert!(weather_params.is_object(), "Parameters should be an object");
        assert!(
            weather_params.get("properties").is_some(),
            "Parameters should have properties"
        );
        assert!(
            weather_params
                .get("properties")
                .unwrap()
                .get("location")
                .is_some(),
            "Location property should exist"
        );
    }

    #[test]
    fn test_tool_definitions_edge_cases() {
        // Set up a client for testing conversion methods
        let client = OllamaClient::new(Some("llama3".to_string())).unwrap();

        // Test with empty tools
        let empty_tools: Vec<ToolDefinition> = vec![];
        let ollama_tools = client.convert_tool_definitions(empty_tools);
        assert!(ollama_tools.is_empty(), "Should produce no tools");

        // Test with a single tool with minimal properties
        let single_tool = vec![ToolDefinition {
            name: "simple".to_string(),
            description: "Simple tool".to_string(),
            parameters: json!({
                "type": "object"
            }),
        }];

        let ollama_tools = client.convert_tool_definitions(single_tool);
        assert_eq!(ollama_tools.len(), 1, "Should produce 1 tool");

        // Check the simple tool
        let simple_tool = &ollama_tools[0];
        assert_eq!(
            simple_tool.tool_type, "function",
            "Tool type should be function"
        );
        assert_eq!(simple_tool.function.name, "simple", "Name should match");
        assert_eq!(
            simple_tool.function.description, "Simple tool",
            "Description should match"
        );

        // Check function parameters
        let simple_params = &simple_tool.function.parameters;
        assert!(simple_params.is_object(), "Parameters should be an object");
        assert_eq!(
            simple_params.get("type").unwrap(),
            "object",
            "Type should be object"
        );
    }
}
