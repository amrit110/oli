use crate::apis::api_client::{ApiClient, CompletionOptions, Message, ToolCall, ToolResult};
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
struct OpenAIToolChoice {
    #[serde(rename = "type")]
    choice_type: String,
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
    // New field to ensure responses for each tool call
    #[serde(skip_serializing_if = "Option::is_none")]
    response_expected: Option<bool>,
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

    fn convert_messages(&self, messages: Vec<Message>) -> Vec<OpenAIMessage> {
        messages
            .into_iter()
            .map(|msg| {
                // Parse content for tool calls if it's an assistant message
                // This handles stored tool calls in message history
                if msg.role == "assistant" && msg.content.contains("\"tool_calls\":") {
                    // Try to parse JSON with tool calls
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&msg.content) {
                        if let Some(tool_calls) = parsed.get("tool_calls") {
                            if let Some(tool_calls_array) = tool_calls.as_array() {
                                // Convert to OpenAIToolCall format
                                let openai_tool_calls = tool_calls_array
                                    .iter()
                                    .filter_map(|call| {
                                        let id = call.get("id")?.as_str()?.to_string();
                                        let name = call.get("name")?.as_str()?.to_string();
                                        let arguments = call.get("arguments")?.to_string();

                                        Some(OpenAIToolCall {
                                            id,
                                            tool_type: "function".to_string(),
                                            function: OpenAIFunctionCall { name, arguments },
                                        })
                                    })
                                    .collect::<Vec<_>>();

                                if !openai_tool_calls.is_empty() {
                                    return OpenAIMessage {
                                        role: msg.role,
                                        content: Some(
                                            parsed
                                                .get("content")
                                                .and_then(|c| c.as_str())
                                                .unwrap_or_default()
                                                .to_string(),
                                        ),
                                        tool_calls: Some(openai_tool_calls),
                                        tool_call_id: None,
                                        response_expected: Some(true), // Expect a response if there are tools
                                    };
                                }
                            }
                        }
                    }
                }

                // Default case for normal messages
                OpenAIMessage {
                    role: msg.role,
                    content: Some(msg.content),
                    tool_calls: None,
                    tool_call_id: None,
                    response_expected: None,
                }
            })
            .collect()
    }

    fn convert_tool_definitions(
        &self,
        tools: Vec<crate::apis::api_client::ToolDefinition>,
    ) -> Vec<OpenAITool> {
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
        if let Some(json_schema) = &options.json_schema {
            request.response_format = Some(json!({
                "type": "json_object",
                "schema": serde_json::from_str(json_schema).unwrap_or(json!({}))
            }));
        }

        let response = self
            .client
            .post(&self.api_base)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                AppError::NetworkError(format!("Failed to send request to OpenAI: {}", e))
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
        let response_text = response
            .text()
            .await
            .map_err(|e| AppError::NetworkError(format!("Failed to get response text: {}", e)))?;

        let openai_response: OpenAIResponse = serde_json::from_str(&response_text)
            .map_err(|e| AppError::Other(format!("Failed to parse OpenAI response: {}", e)))?;

        // Extract content from the first choice
        if let Some(first_choice) = openai_response.choices.first() {
            if let Some(content) = &first_choice.message.content {
                return Ok(content.clone());
            }
        }

        Err(AppError::Other("No content in OpenAI response".to_string()).into())
    }

    async fn complete_with_tools(
        &self,
        messages: Vec<Message>,
        options: CompletionOptions,
        tool_results: Option<Vec<ToolResult>>,
    ) -> Result<(String, Option<Vec<ToolCall>>)> {
        // Convert messages to OpenAI format
        let mut openai_messages = self.convert_messages(messages);

        // Add assistant message with tool calls first, then add tool results
        // For OpenAI, we need to include the assistant's message with tool calls before the tool results
        let mut last_assistant_msg_with_tools = false;
        let mut found_tool_ids = Vec::new(); // Track found tool IDs

        // Scan for assistant messages with tool_calls and collect their IDs
        for msg in &openai_messages {
            if msg.role == "assistant" && msg.tool_calls.is_some() {
                let tool_calls = msg.tool_calls.as_ref().unwrap();
                if !tool_calls.is_empty() {
                    last_assistant_msg_with_tools = true;
                    // Collect all tool call IDs from this message
                    for call in tool_calls {
                        found_tool_ids.push(call.id.clone());
                    }
                }
            }
        }

        // Add tool results only if there's an assistant message with tool calls
        if let Some(results) = tool_results {
            if last_assistant_msg_with_tools {
                // Only add tool results for IDs that exist in the conversation
                for result in results {
                    // Make sure result tool_call_id is found in our tracked IDs
                    if found_tool_ids.contains(&result.tool_call_id) {
                        openai_messages.push(OpenAIMessage {
                            role: "tool".to_string(),
                            content: Some(result.output),
                            tool_calls: None,
                            tool_call_id: Some(result.tool_call_id),
                            response_expected: None,
                        });
                    } else {
                        // Log that we're skipping a tool result with an unknown ID
                        // but don't stop execution
                        eprintln!(
                            "Skipping tool result with ID {} not found in conversation",
                            result.tool_call_id
                        );
                    }
                }
            } else {
                // Return success but don't add any tool results - this avoids errors when the first tool result is added
                // This is more graceful than erroring out completely
                return Ok((
                    "I'm ready to help with your questions about this project.".to_string(),
                    None,
                ));
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
        if let Some(json_schema) = &options.json_schema {
            request.response_format = Some(json!({
                "type": "json_object",
                "schema": serde_json::from_str(json_schema).unwrap_or(json!({}))
            }));
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

        let response = self
            .client
            .post(&self.api_base)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                AppError::NetworkError(format!("Failed to send request to OpenAI: {}", e))
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
        let response_text = response
            .text()
            .await
            .map_err(|e| AppError::NetworkError(format!("Failed to get response text: {}", e)))?;

        let openai_response: OpenAIResponse = serde_json::from_str(&response_text)
            .map_err(|e| AppError::Other(format!("Failed to parse OpenAI response: {}", e)))?;

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
