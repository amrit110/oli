use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: String) -> Self {
        Self {
            role: "system".to_string(),
            content,
        }
    }

    pub fn user(content: String) -> Self {
        Self {
            role: "user".to_string(),
            content,
        }
    }

    pub fn assistant(content: String) -> Self {
        Self {
            role: "assistant".to_string(),
            content,
        }
    }
}

/// Manages the conversation session with history of messages
#[derive(Debug, Clone)]
pub struct SessionManager {
    /// History of messages for the current session
    pub messages: Vec<Message>,
    /// Maximum number of messages to keep in the session
    pub max_messages: usize,
    /// System message to prepend to all conversations
    pub system_message: Option<Message>,
}

impl Default for SessionManager {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            max_messages: 100,
            system_message: None,
        }
    }
}

impl SessionManager {
    /// Create a new session manager with a specific message capacity
    pub fn new(max_messages: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_messages,
            system_message: None,
        }
    }

    /// Add a system message that will be prepended to the conversation
    pub fn with_system_message(mut self, content: String) -> Self {
        self.system_message = Some(Message::system(content));
        self
    }

    /// Add a user message to the conversation
    pub fn add_user_message(&mut self, content: String) {
        self.add_message(Message::user(content));
    }

    /// Add an assistant message to the conversation
    pub fn add_assistant_message(&mut self, content: String) {
        self.add_message(Message::assistant(content));
    }

    /// Add a message to the conversation
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
        self.trim_if_needed();
    }

    /// Replace all messages with a single summary message
    pub fn replace_with_summary(&mut self, summary: String) {
        self.messages.clear();
        self.add_message(Message::system(format!(
            "Previous conversation summary: {}",
            summary
        )));
    }

    /// Get all messages for the API call, including the system message if present
    pub fn get_messages_for_api(&self) -> Vec<Message> {
        let mut api_messages = Vec::new();

        // Add system message if present
        if let Some(sys_message) = &self.system_message {
            api_messages.push(sys_message.clone());
        }

        // Add conversation messages
        api_messages.extend(self.messages.clone());

        api_messages
    }

    /// Clear all messages in the session
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Get the current number of messages
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Trim messages if the count exceeds max_messages
    fn trim_if_needed(&mut self) {
        if self.messages.len() > self.max_messages {
            let to_remove = self.messages.len() - self.max_messages;
            self.messages.drain(0..to_remove);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: Option<String>, // Required for OpenAI to map tool results back to calls
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub output: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub json_schema: Option<String>,
    pub require_tool_use: bool,
}

impl Default for CompletionOptions {
    fn default() -> Self {
        Self {
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_tokens: Some(2048),
            tools: None,
            json_schema: None,
            require_tool_use: false,
        }
    }
}

// This trait cannot be made into a dyn trait because it has async methods
#[async_trait::async_trait]
pub trait ApiClient: Send + Sync {
    // Basic completion without tool usage
    #[allow(dead_code)]
    async fn complete(&self, messages: Vec<Message>, options: CompletionOptions) -> Result<String>;

    async fn complete_with_tools(
        &self,
        messages: Vec<Message>,
        options: CompletionOptions,
        tool_results: Option<Vec<ToolResult>>,
    ) -> Result<(String, Option<Vec<ToolCall>>)>;
}

// Instead of using a trait object, we'll use an enum to handle different providers
#[derive(Clone)]
pub enum ApiClientEnum {
    Anthropic(Arc<crate::apis::anthropic::AnthropicClient>),
    OpenAI(Arc<crate::apis::openai::OpenAIClient>),
    Ollama(Arc<crate::apis::ollama::OllamaClient>),
    Gemini(Arc<crate::apis::gemini::GeminiClient>),
    CustomMock(Arc<dyn ApiClient>),
}

impl ApiClientEnum {
    #[allow(dead_code)]
    pub async fn complete(
        &self,
        messages: Vec<Message>,
        options: CompletionOptions,
    ) -> Result<String> {
        match self {
            Self::Anthropic(client) => client.complete(messages, options).await,
            Self::OpenAI(client) => client.complete(messages, options).await,
            Self::Ollama(client) => client.complete(messages, options).await,
            Self::Gemini(client) => client.complete(messages, options).await,
            Self::CustomMock(client) => client.complete(messages, options).await,
        }
    }

    pub async fn complete_with_tools(
        &self,
        messages: Vec<Message>,
        options: CompletionOptions,
        tool_results: Option<Vec<ToolResult>>,
    ) -> Result<(String, Option<Vec<ToolCall>>)> {
        match self {
            Self::Anthropic(client) => {
                client
                    .complete_with_tools(messages, options, tool_results)
                    .await
            }
            Self::OpenAI(client) => {
                client
                    .complete_with_tools(messages, options, tool_results)
                    .await
            }
            Self::Ollama(client) => {
                client
                    .complete_with_tools(messages, options, tool_results)
                    .await
            }
            Self::Gemini(client) => {
                client
                    .complete_with_tools(messages, options, tool_results)
                    .await
            }
            Self::CustomMock(client) => {
                client
                    .complete_with_tools(messages, options, tool_results)
                    .await
            }
        }
    }

    pub fn custom_for_testing(client: Arc<dyn ApiClient>) -> Self {
        Self::CustomMock(client)
    }
}

pub type DynApiClient = ApiClientEnum;
