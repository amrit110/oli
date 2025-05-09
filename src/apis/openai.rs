use crate::apis::api_client::{
    ApiClient, CompletionOptions, Message, ToolCall, ToolDefinition, ToolResult,
};
use crate::app::logger::{format_log_with_color, LogLevel};
use crate::errors::AppError;
use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::Client as ReqwestClient;
use serde::{Deserialize, Serialize};
use serde_json::{self, json, Value};
use std::env;

// OpenAI API Types
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAIFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAIFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIResponseChoice {
    index: usize,
    message: OpenAIMessage,
    finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<OpenAIResponseChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Value>,
}

pub struct OpenAIClient {
    client: ReqwestClient,
    model: String,
    api_base: String,
}

// Helper methods
impl OpenAIClient {
    /// Returns the model name being used by this client
    ///
    /// Primarily used for testing purposes.
    #[cfg(test)]
    pub(crate) fn get_model_name(&self) -> &str {
        &self.model
    }
}

impl OpenAIClient {
    pub fn new(model: Option<String>) -> Result<Self> {
        // Try to get API key from environment
        let api_key =
            env::var("OPENAI_API_KEY").context("OPENAI_API_KEY environment variable not set")?;

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

        let client = ReqwestClient::builder().default_headers(headers).build()?;

        // Default to GPT-4o as the latest model with tooling capabilities
        let model = model.unwrap_or_else(|| "gpt-4o".to_string());

        Ok(Self {
            client,
            model,
            api_base: "https://api.openai.com/v1/chat/completions".to_string(),
        })
    }

    /// Converts internal message format to OpenAI's message format
    ///
    /// This method converts each message to OpenAI's format with appropriate
    /// role and content fields.
    fn convert_messages(&self, messages: Vec<Message>) -> Vec<OpenAIMessage> {
        messages
            .into_iter()
            .map(|msg| {
                // Convert standard messages
                OpenAIMessage {
                    role: msg.role,
                    content: Some(msg.content),
                    tool_calls: None,
                    tool_call_id: None,
                }
            })
            .collect()
    }

    /// Converts internal tool definitions to OpenAI's format
    ///
    /// This method converts tool definitions to OpenAI's function format with
    /// appropriate name, description, and parameters.
    fn convert_tool_definitions(&self, tools: Vec<ToolDefinition>) -> Vec<OpenAITool> {
        tools
            .into_iter()
            .map(|tool| OpenAITool {
                tool_type: "function".to_string(),
                function: OpenAIFunction {
                    name: tool.name,
                    description: tool.description,
                    parameters: tool.parameters,
                },
            })
            .collect()
    }
}

#[async_trait]
impl ApiClient for OpenAIClient {
    async fn complete(&self, messages: Vec<Message>, options: CompletionOptions) -> Result<String> {
        let openai_messages = self.convert_messages(messages);

        let mut request = OpenAIRequest {
            model: self.model.clone(),
            messages: openai_messages,
            max_tokens: options.max_tokens,
            temperature: options.temperature,
            top_p: options.top_p,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        // Add structured output format if specified in options
        if let Some(_json_schema) = &options.json_schema {
            request.response_format = Some(json!({
                "type": "json_object"
            }));
        }

        eprintln!(
            "{}",
            format_log_with_color(
                LogLevel::Debug,
                &format!("Sending request to OpenAI API with model: {}", self.model)
            )
        );

        let response = self
            .client
            .post(&self.api_base)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                let error_msg = format!("Failed to send request to OpenAI: {}", e);
                eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
                AppError::NetworkError(error_msg)
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AppError::NetworkError(format!(
                "OpenAI API error: {} - {}",
                status, error_text
            ))
            .into());
        }

        // Parse response
        let response_text = response.text().await.map_err(|e| {
            let error_msg = format!("Failed to get response text: {}", e);
            eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
            AppError::NetworkError(error_msg)
        })?;

        eprintln!(
            "{}",
            format_log_with_color(
                LogLevel::Debug,
                &format!(
                    "OpenAI API response received: {} bytes",
                    response_text.len()
                )
            )
        );

        let openai_response: OpenAIResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                let error_msg = format!("Failed to parse OpenAI response: {}", e);
                eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
                AppError::Other(error_msg)
            })?;

        // Extract content from the first choice
        if let Some(first_choice) = openai_response.choices.first() {
            if let Some(content) = &first_choice.message.content {
                return Ok(content.clone());
            }
        }

        let error_msg = "No content in OpenAI response".to_string();
        eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
        Err(AppError::LLMError(error_msg).into())
    }

    async fn complete_with_tools(
        &self,
        messages: Vec<Message>,
        options: CompletionOptions,
        tool_results: Option<Vec<ToolResult>>,
    ) -> Result<(String, Option<Vec<ToolCall>>)> {
        // Convert messages to OpenAI format
        let mut openai_messages = self.convert_messages(messages);

        // Track tool calls that need responses
        let mut pending_tool_calls = Vec::new();

        // First pass: identify all tool calls that need responses
        for msg in &openai_messages {
            if msg.role == "assistant" && msg.tool_calls.is_some() {
                if let Some(tool_calls) = &msg.tool_calls {
                    for call in tool_calls {
                        pending_tool_calls.push(call.id.clone());
                    }
                }
            }
        }

        // Second pass: remove tool call IDs that already have responses
        for msg in &openai_messages {
            if msg.role == "tool" && msg.tool_call_id.is_some() {
                if let Some(tool_call_id) = &msg.tool_call_id {
                    pending_tool_calls.retain(|id| id != tool_call_id);
                }
            }
        }

        // Add tool results for any pending tool calls
        if let Some(results) = &tool_results {
            let result_map: std::collections::HashMap<String, String> = results
                .iter()
                .map(|r| (r.tool_call_id.clone(), r.output.clone()))
                .collect();

            // Add responses for any pending tool calls
            for tool_id in &pending_tool_calls {
                if let Some(output) = result_map.get(tool_id) {
                    openai_messages.push(OpenAIMessage {
                        role: "tool".to_string(),
                        content: Some(output.clone()),
                        tool_calls: None,
                        tool_call_id: Some(tool_id.clone()),
                    });
                } else {
                    // For any tool call without a provided result, add a default response
                    // This is crucial for OpenAI - every tool call must have a response
                    openai_messages.push(OpenAIMessage {
                        role: "tool".to_string(),
                        content: Some(
                            "Tool execution completed without detailed results.".to_string(),
                        ),
                        tool_calls: None,
                        tool_call_id: Some(tool_id.clone()),
                    });
                }
            }
        } else if !pending_tool_calls.is_empty() {
            // If we have pending tool calls but no results were provided,
            // we need to add default responses for all pending tool calls
            for tool_id in &pending_tool_calls {
                openai_messages.push(OpenAIMessage {
                    role: "tool".to_string(),
                    content: Some("Tool execution completed without detailed results.".to_string()),
                    tool_calls: None,
                    tool_call_id: Some(tool_id.clone()),
                });
            }
        }

        let mut request = OpenAIRequest {
            model: self.model.clone(),
            messages: openai_messages,
            max_tokens: options.max_tokens,
            temperature: options.temperature,
            top_p: options.top_p,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        // Add structured output format if specified in options
        if let Some(_json_schema) = &options.json_schema {
            request.response_format = Some(json!({
                "type": "json_object"
            }));

            // Ensure at least one message contains the word "json" when using json_object response format
            let has_json_keyword = request.messages.iter().any(|msg| {
                msg.content
                    .as_ref()
                    .is_some_and(|content| content.to_lowercase().contains("json"))
            });

            if !has_json_keyword && !request.messages.is_empty() {
                // Add "json" to the user's last message if it doesn't already contain it
                if let Some(last_user_msg) = request
                    .messages
                    .iter_mut()
                    .rev()
                    .find(|msg| msg.role == "user")
                {
                    if let Some(content) = &mut last_user_msg.content {
                        *content = format!("{} (Please provide the response as JSON)", content);
                    }
                }
            }
        }

        // Add tools if they exist
        if let Some(tools) = options.tools {
            let converted_tools = self.convert_tool_definitions(tools);
            request.tools = Some(converted_tools);

            // Set tool_choice based on option
            request.tool_choice = if options.require_tool_use {
                Some("required".to_string())
            } else {
                Some("auto".to_string())
            };
        }

        eprintln!(
            "{}",
            format_log_with_color(
                LogLevel::Debug,
                &format!("Sending request to OpenAI API with model: {}", self.model)
            )
        );

        let response = self
            .client
            .post(&self.api_base)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                let error_msg = format!("Failed to send request to OpenAI: {}", e);
                eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
                AppError::NetworkError(error_msg)
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AppError::NetworkError(format!(
                "OpenAI API error: {} - {}",
                status, error_text
            ))
            .into());
        }

        // Parse response
        let response_text = response.text().await.map_err(|e| {
            let error_msg = format!("Failed to get response text: {}", e);
            eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
            AppError::NetworkError(error_msg)
        })?;

        eprintln!(
            "{}",
            format_log_with_color(
                LogLevel::Debug,
                &format!(
                    "OpenAI API response received: {} bytes",
                    response_text.len()
                )
            )
        );

        let openai_response: OpenAIResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                let error_msg = format!("Failed to parse OpenAI response: {}", e);
                eprintln!("{}", format_log_with_color(LogLevel::Error, &error_msg));
                AppError::Other(error_msg)
            })?;

        // Extract content and tool calls from the first choice
        if let Some(first_choice) = openai_response.choices.first() {
            let content = first_choice.message.content.clone().unwrap_or_default();

            // Extract tool calls if present
            let tool_calls = if let Some(openai_tool_calls) = &first_choice.message.tool_calls {
                if openai_tool_calls.is_empty() {
                    None
                } else {
                    let calls = openai_tool_calls
                        .iter()
                        .map(|call| {
                            // Parse arguments as JSON
                            let arguments_result =
                                serde_json::from_str::<Value>(&call.function.arguments);
                            let arguments = match arguments_result {
                                Ok(args) => args,
                                Err(_) => json!({}),
                            };

                            // Create a tool call with OpenAI's required format
                            ToolCall {
                                id: Some(call.id.clone()), // Important for tool results later
                                name: call.function.name.clone(),
                                arguments,
                            }
                        })
                        .collect::<Vec<_>>();

                    if calls.is_empty() {
                        None
                    } else {
                        Some(calls)
                    }
                }
            } else {
                None
            };

            return Ok((content, tool_calls));
        }

        Ok((String::new(), None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apis::api_client::{Message, ToolDefinition};
    use serde_json::json;

    #[test]
    fn test_openai_model_name() {
        // Test that the default model name is correct when providing None
        // This doesn't make API calls, just tests the client setup logic
        let api_key = "test_api_key".to_string();
        let client = OpenAIClient::with_api_key(api_key, None).unwrap();

        // Verify the model name is the expected default
        assert_eq!(
            client.get_model_name(),
            "gpt-4o",
            "Default model name should be gpt-4o"
        );
    }

    #[test]
    fn test_openai_with_custom_model() {
        // Test that the custom model name is used correctly
        let api_key = "test_api_key".to_string();
        let model_name = "gpt-4-turbo".to_string();
        let client = OpenAIClient::with_api_key(api_key, Some(model_name.clone())).unwrap();

        // Verify the custom model name is used
        assert_eq!(
            client.get_model_name(),
            model_name,
            "Custom model name should be used"
        );
    }

    #[test]
    fn test_message_conversion() {
        // Set up a client for testing conversion methods
        let api_key = "test_api_key".to_string();
        let client = OpenAIClient::with_api_key(api_key, None).unwrap();

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
        let openai_messages = client.convert_messages(messages);

        // Verify messages are converted correctly
        assert_eq!(openai_messages.len(), 3, "Should have 3 messages");

        // Check system message
        let system_msg = &openai_messages[0];
        assert_eq!(
            system_msg.role, "system",
            "First message should be a system message"
        );
        assert_eq!(
            system_msg.content.as_ref().unwrap(),
            "You are a helpful assistant.",
            "Content should match"
        );
        assert!(
            system_msg.tool_calls.is_none(),
            "System message should not have tool calls"
        );

        // Check user message
        let user_msg = &openai_messages[1];
        assert_eq!(
            user_msg.role, "user",
            "Second message should be a user message"
        );
        assert_eq!(
            user_msg.content.as_ref().unwrap(),
            "Hello",
            "Content should match"
        );
        assert!(
            user_msg.tool_calls.is_none(),
            "User message should not have tool calls"
        );

        // Check assistant message
        let assistant_msg = &openai_messages[2];
        assert_eq!(
            assistant_msg.role, "assistant",
            "Third message should be an assistant message"
        );
        assert_eq!(
            assistant_msg.content.as_ref().unwrap(),
            "Hi there! How can I help you today?",
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
        let api_key = "test_api_key".to_string();
        let client = OpenAIClient::with_api_key(api_key, None).unwrap();

        // Test with empty messages
        let empty_messages: Vec<Message> = vec![];
        let openai_messages = client.convert_messages(empty_messages);
        assert!(openai_messages.is_empty(), "Should produce no messages");

        // Test with a single message
        let single_message = vec![Message {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];

        let openai_messages = client.convert_messages(single_message);
        assert_eq!(openai_messages.len(), 1, "Should produce 1 message");
        assert_eq!(openai_messages[0].role, "user", "Should be a user message");
        assert_eq!(
            openai_messages[0].content.as_ref().unwrap(),
            "Hello",
            "Content should match"
        );
    }

    #[test]
    fn test_tool_definitions_conversion() {
        // Set up a client for testing conversion methods
        let api_key = "test_api_key".to_string();
        let client = OpenAIClient::with_api_key(api_key, None).unwrap();

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
        let openai_tools = client.convert_tool_definitions(tools);

        // Verify tools are converted correctly
        assert_eq!(openai_tools.len(), 2, "Should have 2 tools");

        // Check first tool
        let calculator_tool = &openai_tools[0];
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
        let weather_tool = &openai_tools[1];
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
        let api_key = "test_api_key".to_string();
        let client = OpenAIClient::with_api_key(api_key, None).unwrap();

        // Test with empty tools
        let empty_tools: Vec<ToolDefinition> = vec![];
        let openai_tools = client.convert_tool_definitions(empty_tools);
        assert!(openai_tools.is_empty(), "Should produce no tools");

        // Test with a single tool with minimal properties
        let single_tool = vec![ToolDefinition {
            name: "simple".to_string(),
            description: "Simple tool".to_string(),
            parameters: json!({
                "type": "object"
            }),
        }];

        let openai_tools = client.convert_tool_definitions(single_tool);
        assert_eq!(openai_tools.len(), 1, "Should produce 1 tool");

        // Check the simple tool
        let simple_tool = &openai_tools[0];
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
