use crate::agent::executor::AgentExecutor;
use crate::apis::anthropic::AnthropicClient;
use crate::apis::api_client::{ApiClientEnum, DynApiClient, Message};
use crate::apis::gemini::GeminiClient;
use crate::apis::ollama::OllamaClient;
use crate::apis::openai::OpenAIClient;
use crate::prompts::DEFAULT_AGENT_PROMPT;
use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Clone)]
pub enum LLMProvider {
    Anthropic,
    OpenAI,
    Ollama,
    Gemini,
}

#[derive(Clone)]
pub struct Agent {
    provider: LLMProvider,
    model: Option<String>,
    api_client: Option<DynApiClient>,
    system_prompt: Option<String>,
    progress_sender: Option<mpsc::Sender<String>>,
    // Store the conversation history
    conversation_history: Vec<crate::apis::api_client::Message>,
}

impl Agent {
    pub fn new(provider: LLMProvider) -> Self {
        Self {
            provider,
            model: None,
            api_client: None,
            system_prompt: None,
            progress_sender: None,
            conversation_history: Vec::new(),
        }
    }

    pub fn new_with_api_key(provider: LLMProvider, api_key: String) -> Self {
        // Create a new agent with the given provider and API key
        // The API key will be used during initialization
        let mut agent = Self::new(provider);
        // Store the API key as the model temporarily
        // It will be handled properly in initialize_with_api_key
        agent.model = Some(api_key);
        agent
    }

    pub fn with_model(mut self, model: String) -> Self {
        self.model = Some(model);
        self
    }

    pub fn with_system_prompt(mut self, prompt: String) -> Self {
        self.system_prompt = Some(prompt);
        self
    }

    pub fn with_progress_sender(mut self, sender: mpsc::Sender<String>) -> Self {
        self.progress_sender = Some(sender);
        self
    }

    pub fn clear_history(&mut self) {
        self.conversation_history.clear();
    }

    /// Add a message to the conversation history
    pub fn add_message(&mut self, message: Message) {
        self.conversation_history.push(message);
    }

    /// Get a clone of the conversation history (for testing)
    pub fn get_conversation_history_for_test(&self) -> Vec<Message> {
        self.conversation_history.clone()
    }

    pub async fn initialize(&mut self) -> Result<()> {
        // Create the API client based on provider and model
        self.api_client = Some(match self.provider {
            LLMProvider::Anthropic => {
                let client = AnthropicClient::new(self.model.clone())?;
                ApiClientEnum::Anthropic(Arc::new(client))
            }
            LLMProvider::OpenAI => {
                let client = OpenAIClient::new(self.model.clone())?;
                ApiClientEnum::OpenAI(Arc::new(client))
            }
            LLMProvider::Ollama => {
                let client = OllamaClient::new(self.model.clone())?;
                ApiClientEnum::Ollama(Arc::new(client))
            }
            LLMProvider::Gemini => {
                let client = GeminiClient::new(self.model.clone())?;
                ApiClientEnum::Gemini(Arc::new(client))
            }
        });

        Ok(())
    }

    pub async fn initialize_with_api_key(&mut self, api_key: String) -> Result<()> {
        // Create the API client based on provider and model, using the provided API key
        self.api_client = Some(match self.provider {
            LLMProvider::Anthropic => {
                let client = AnthropicClient::with_api_key(api_key, self.model.clone())?;
                ApiClientEnum::Anthropic(Arc::new(client))
            }
            LLMProvider::OpenAI => {
                let client = OpenAIClient::with_api_key(api_key, self.model.clone())?;
                ApiClientEnum::OpenAI(Arc::new(client))
            }
            LLMProvider::Ollama => {
                // For Ollama, we always use the local URL
                // API keys don't apply to local Ollama instances
                let client = OllamaClient::new(self.model.clone())?;
                ApiClientEnum::Ollama(Arc::new(client))
            }
            LLMProvider::Gemini => {
                let client = GeminiClient::with_api_key(api_key, self.model.clone())?;
                ApiClientEnum::Gemini(Arc::new(client))
            }
        });

        Ok(())
    }

    pub async fn execute(&self, query: &str) -> Result<String> {
        let api_client = self
            .api_client
            .as_ref()
            .context("Agent not initialized. Call initialize() first.")?;

        // Create and configure executor with persisted conversation history
        let mut executor = AgentExecutor::new(api_client.clone());

        // Add existing conversation history if any
        if !self.conversation_history.is_empty() {
            executor.set_conversation_history(self.conversation_history.clone());
        }

        // Log the conversation history we're passing to the executor only when debug is explicitly enabled
        let is_debug_mode = std::env::var("RUST_LOG")
            .map(|v| v.contains("debug"))
            .unwrap_or(false);

        if is_debug_mode {
            if let Some(progress_sender) = &self.progress_sender {
                let _ = progress_sender.try_send(format!(
                    "[debug] Agent execute with history: {} messages",
                    self.conversation_history.len()
                ));
                for (i, msg) in self.conversation_history.iter().enumerate() {
                    let _ = progress_sender.try_send(format!(
                        "[debug]   History message {}: role={}, preview={}",
                        i,
                        msg.role,
                        if msg.content.len() > 30 {
                            format!("{}...", &msg.content[..30])
                        } else {
                            msg.content.clone()
                        }
                    ));
                }
            }
        }

        // Add progress sender if available
        if let Some(sender) = &self.progress_sender {
            executor = executor.with_progress_sender(sender.clone());
        }

        // Always preserve system message at the beginning - if it doesn't exist
        let has_system_message = self
            .conversation_history
            .iter()
            .any(|msg| msg.role == "system");

        // Add system prompt if it doesn't exist in history
        if !has_system_message {
            // Add system prompt if available
            if let Some(system_prompt) = &self.system_prompt {
                executor.add_system_message(system_prompt.clone());
            } else {
                // Use default system prompt
                executor.add_system_message(DEFAULT_AGENT_PROMPT.to_string());
            }
        }

        // Add the original user query
        executor.add_user_message(query.to_string());

        // Execute and get result
        let result = executor.execute().await?;

        // Save updated conversation history for future calls
        // We need to make sure we preserve the system message in the history
        if let Some(mutable_self) = unsafe { (self as *const Self as *mut Self).as_mut() } {
            // Get updated history from executor
            let mut updated_history = executor.get_conversation_history();

            // Make sure we have a system message, without it conversation history won't work properly
            let has_system_in_updated = updated_history.iter().any(|msg| msg.role == "system");

            // Always ensure we have a system message
            if !has_system_in_updated {
                // Get system message from original history or from system_prompt
                let system_content = mutable_self
                    .conversation_history
                    .iter()
                    .find(|msg| msg.role == "system")
                    .map(|msg| msg.content.clone())
                    .or_else(|| mutable_self.system_prompt.clone())
                    .unwrap_or_else(|| DEFAULT_AGENT_PROMPT.to_string());

                // Insert system message at the beginning
                updated_history.insert(0, Message::system(system_content));
            }

            // Remove any duplicate system messages that might have been added
            let mut seen_system = false;
            updated_history.retain(|msg| {
                if msg.role == "system" {
                    if seen_system {
                        return false; // Remove duplicate system messages
                    }
                    seen_system = true;
                }
                true
            });

            // Make sure the system message is at the beginning
            updated_history.sort_by(|a, b| {
                if a.role == "system" {
                    std::cmp::Ordering::Less
                } else if b.role == "system" {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            });

            // Update the history
            mutable_self.conversation_history = updated_history;

            // Debug: Log the updated conversation history only when debug is explicitly enabled
            let is_debug_mode = std::env::var("RUST_LOG")
                .map(|v| v.contains("debug"))
                .unwrap_or(false);

            if is_debug_mode {
                if let Some(progress_sender) = &self.progress_sender {
                    let _ = progress_sender.try_send(format!(
                        "[debug] Updated conversation history: {} messages",
                        mutable_self.conversation_history.len()
                    ));
                    for (i, msg) in mutable_self.conversation_history.iter().enumerate() {
                        let _ = progress_sender.try_send(format!(
                            "[debug]   Updated message {}: role={}, preview={}",
                            i,
                            msg.role,
                            if msg.content.len() > 30 {
                                format!("{}...", &msg.content[..30])
                            } else {
                                msg.content.clone()
                            }
                        ));
                    }
                }
            }
        }

        Ok(result)
    }
}
