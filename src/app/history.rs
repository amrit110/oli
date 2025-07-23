use crate::agent::core::Agent;
use crate::apis::api_client::Message;
use crate::app::core::{App, AppState};
use crate::prompts::CONVERSATION_SUMMARY_PROMPT;
use anyhow::Result;
use std::time::Instant;

/// Message content threshold before considering summarization (in chars)
const DEFAULT_SUMMARIZATION_CHAR_THRESHOLD: usize = 1000000;
/// Message count threshold before considering summarization
const DEFAULT_SUMMARIZATION_COUNT_THRESHOLD: usize = 1000;
/// Maximum number of messages to keep unsummarized (recent history)
const DEFAULT_KEEP_RECENT_COUNT: usize = 20;

#[derive(Clone)]
/// Represents a conversation summary
pub struct ConversationSummary {
    /// The summarized content
    pub content: String,
    /// When the summary was created
    pub created_at: Instant,
    /// Number of messages summarized
    pub messages_count: usize,
    /// Original character count that was summarized
    pub original_chars: usize,
}

impl ConversationSummary {
    pub fn new(content: String, messages_count: usize, original_chars: usize) -> Self {
        Self {
            content,
            created_at: Instant::now(),
            messages_count,
            original_chars,
        }
    }
}

/// Context compression management trait for the application
pub trait ContextCompressor {
    /// Generate a summary of the conversation history
    fn compress_context(&mut self) -> Result<()>;

    /// Check if conversation should be summarized based on thresholds
    fn should_compress(&self) -> bool;

    /// Get the total character count of conversation history
    fn conversation_char_count(&self) -> usize;

    /// Get summaries count
    fn summary_count(&self) -> usize;

    /// Clear all summaries and history
    fn clear_history(&mut self);

    /// Convert display messages to session messages
    fn display_to_session_messages(&self, display_messages: &[String]) -> Vec<Message>;

    /// Convert session messages to display messages
    fn session_to_display_messages(&self, session_messages: &[Message]) -> Vec<String>;
}

impl ContextCompressor for App {
    fn compress_context(&mut self) -> Result<()> {
        // Don't summarize if no messages
        if self.messages.is_empty() {
            return Ok(());
        }

        // Check if we have an agent for summarization
        let agent = match &self.agent {
            Some(agent) => agent.clone(),
            None => return Err(anyhow::anyhow!("No agent available for summarization")),
        };

        // Keep the most recent messages unsummarized
        let keep_recent = DEFAULT_KEEP_RECENT_COUNT.min(self.messages.len());
        let to_summarize = self.messages.len().saturating_sub(keep_recent);

        // If nothing to summarize, just return
        if to_summarize == 0 {
            return Ok(());
        }

        // Get the messages to summarize
        let messages_to_summarize = self.messages[0..to_summarize].join("\n");
        let messages_chars = messages_to_summarize.len();

        // Show a message that we're summarizing
        self.messages
            .push("[wait] ⚪ Summarizing conversation history...".into());

        // Generate the summary using the agent
        let summary = self.generate_summary_with_agent(&agent, &messages_to_summarize)?;

        // Create a new summary record
        let summary_record =
            ConversationSummary::new(summary.clone(), to_summarize, messages_chars);

        // Add the summary to the list
        self.conversation_summaries.push(summary_record);

        // Remove the summarized messages
        self.messages.drain(0..to_summarize);

        // Add the summary marker to the message list
        self.messages.insert(
            0,
            format!("💬 [CONVERSATION SUMMARY]\n{summary}\n[END SUMMARY]"),
        );

        // Update the session manager if it exists
        // First collect the messages to convert
        let messages_to_keep = self.messages[to_summarize..].to_vec();

        // Convert display messages to API messages format
        let session_messages = self.display_to_session_messages(&messages_to_keep);

        // Update the session manager if it exists
        if let Some(session) = &mut self.session_manager {
            // Replace the session with the summary and recent messages
            session.replace_with_summary(summary.clone());

            // Add the recent messages that weren't summarized
            for msg in session_messages {
                session.add_message(msg.clone());
            }
        }

        // Add a notification
        self.messages.push(format!(
            "[success] ⏺ Summarized {to_summarize} messages ({messages_chars} chars)"
        ));

        // No auto-scroll needed in backend-only mode

        Ok(())
    }

    fn should_compress(&self) -> bool {
        // Don't summarize in non-chat state
        if self.state != AppState::Chat {
            return false;
        }

        // Check both message count and character count thresholds
        let message_count = self.messages.len();
        let char_count = self.conversation_char_count();

        // Also check the session manager if available
        let session_count = self
            .session_manager
            .as_ref()
            .map_or(0, |s| s.message_count());

        message_count > DEFAULT_SUMMARIZATION_COUNT_THRESHOLD
            || char_count > DEFAULT_SUMMARIZATION_CHAR_THRESHOLD
            || session_count > DEFAULT_SUMMARIZATION_COUNT_THRESHOLD
    }

    fn conversation_char_count(&self) -> usize {
        self.messages.iter().map(|m| m.len()).sum()
    }

    fn summary_count(&self) -> usize {
        self.conversation_summaries.len()
    }

    fn clear_history(&mut self) {
        self.messages.clear();
        self.conversation_summaries.clear();

        // No scrolling needed in backend-only mode

        // Also clear agent's conversation history if it exists
        if let Some(agent) = &mut self.agent {
            agent.clear_history();
        }

        // Clear session manager if it exists
        if let Some(session) = &mut self.session_manager {
            session.clear();
        }

        // Notify clients that history was cleared
        self.messages.push("[info] Chat history cleared".into());
    }

    fn display_to_session_messages(&self, display_messages: &[String]) -> Vec<Message> {
        let mut session_messages = Vec::new();
        let mut current_role = "user";

        for msg in display_messages {
            // Try to determine the role based on common message patterns
            if msg.starts_with("[user]") || msg.starts_with("User:") {
                current_role = "user";
                let content = msg
                    .replace("[user]", "")
                    .replace("User:", "")
                    .trim()
                    .to_string();
                session_messages.push(Message::user(content));
            } else if msg.starts_with("[assistant]") || msg.starts_with("Assistant:") {
                current_role = "assistant";
                let content = msg
                    .replace("[assistant]", "")
                    .replace("Assistant:", "")
                    .trim()
                    .to_string();
                session_messages.push(Message::assistant(content));
            } else if msg.starts_with("[system]") || msg.starts_with("System:") {
                current_role = "system";
                let content = msg
                    .replace("[system]", "")
                    .replace("System:", "")
                    .trim()
                    .to_string();
                session_messages.push(Message::system(content));
            } else if !msg.starts_with("[wait]")
                && !msg.starts_with("[success]")
                && !msg.starts_with("[info]")
            {
                // For unmarked messages, use the current role context
                match current_role {
                    "user" => session_messages.push(Message::user(msg.clone())),
                    "assistant" => session_messages.push(Message::assistant(msg.clone())),
                    "system" => session_messages.push(Message::system(msg.clone())),
                    _ => session_messages.push(Message::user(msg.clone())),
                }
            }
        }

        session_messages
    }

    fn session_to_display_messages(&self, session_messages: &[Message]) -> Vec<String> {
        session_messages
            .iter()
            .map(|msg| match msg.role.as_str() {
                "user" => format!("[user] {}", msg.content),
                "assistant" => format!("[assistant] {}", msg.content),
                "system" => format!("[system] {}", msg.content),
                _ => msg.content.clone(),
            })
            .collect()
    }
}

impl App {
    /// Internal method to generate a summary with the agent
    fn generate_summary_with_agent(&mut self, agent: &Agent, content: &str) -> Result<String> {
        // Create a tokio runtime if needed
        let runtime = match &self.tokio_runtime {
            Some(rt) => rt,
            None => return Err(anyhow::anyhow!("Async runtime not available")),
        };

        // Create a cloned agent to avoid borrowing issues
        let agent_clone = agent.clone();

        // Copy the content for the async block
        let content_to_summarize = content.to_string();

        // Define the summarization prompt
        let prompt = format!("{CONVERSATION_SUMMARY_PROMPT}{content_to_summarize}");

        // Execute the summarization
        let result = runtime.block_on(async { agent_clone.execute(&prompt).await })?;

        Ok(result)
    }
}
