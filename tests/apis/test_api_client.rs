//! Unit tests for the API client module

use oli_server::apis::api_client::{
    CompletionOptions, Message, SessionManager, ToolCall, ToolDefinition, ToolResult,
};
use serde_json::json;

/// Tests for the Message struct
#[test]
fn test_message_creation() {
    // Test system message creation
    let system_msg = Message::system("Test system message".to_string());
    assert_eq!(system_msg.role, "system");
    assert_eq!(system_msg.content, "Test system message");

    // Test user message creation
    let user_msg = Message::user("Test user message".to_string());
    assert_eq!(user_msg.role, "user");
    assert_eq!(user_msg.content, "Test user message");

    // Test assistant message creation
    let assistant_msg = Message::assistant("Test assistant message".to_string());
    assert_eq!(assistant_msg.role, "assistant");
    assert_eq!(assistant_msg.content, "Test assistant message");
}

/// Tests for tool structures
#[test]
fn test_tool_structures() {
    // Create a sample tool definition
    let tool_def = ToolDefinition {
        name: "TestTool".to_string(),
        description: "A test tool".to_string(),
        parameters: json!({
            "type": "object",
            "required": ["test_param"],
            "properties": {
                "test_param": {
                    "type": "string",
                    "description": "A test parameter"
                }
            }
        }),
    };

    // Verify tool definition
    assert_eq!(tool_def.name, "TestTool");
    assert_eq!(tool_def.description, "A test tool");

    // Create a tool call with the updated structure
    let tool_call = ToolCall {
        id: Some("call_123".to_string()),
        name: "TestTool".to_string(),
        arguments: json!({
            "test_param": "test_value"
        }),
    };

    // Verify tool call
    assert_eq!(tool_call.id, Some("call_123".to_string()));
    assert_eq!(tool_call.name, "TestTool");
    assert_eq!(
        tool_call
            .arguments
            .get("test_param")
            .unwrap()
            .as_str()
            .unwrap(),
        "test_value"
    );

    // Create a tool result with the updated structure
    let tool_result = ToolResult {
        tool_call_id: "call_123".to_string(),
        output: "Tool execution result".to_string(),
    };

    // Verify tool result
    assert_eq!(tool_result.tool_call_id, "call_123");
    assert_eq!(tool_result.output, "Tool execution result");
}

/// Tests for completion options
#[test]
fn test_completion_options() {
    // Create default completion options
    let options = CompletionOptions::default();

    // Default values should be set
    assert!(options.max_tokens.unwrap() > 0);
    assert!(options.temperature.unwrap() >= 0.0 && options.temperature.unwrap() <= 1.0);

    // Create custom options
    let custom_options = CompletionOptions {
        max_tokens: Some(1000),
        temperature: Some(0.8),
        tools: Some(vec![ToolDefinition {
            name: "TestTool".to_string(),
            description: "A test tool".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {}
            }),
        }]),
        require_tool_use: false,
        ..Default::default()
    };

    // Verify custom values
    assert_eq!(custom_options.max_tokens, Some(1000));
    assert_eq!(custom_options.temperature, Some(0.8));
    assert!(!custom_options.require_tool_use);
    assert_eq!(
        custom_options.tools.as_ref().unwrap()[0].name,
        "TestTool".to_string()
    );
}

/// Tests for session manager
#[cfg(test)]
pub mod session_manager_tests {
    use super::*;

    #[test]
    fn test_session_manager_creation() {
        // Create a new session manager with default max messages (100)
        let session_manager = SessionManager::new(100);

        // Initial state should be empty
        assert!(session_manager.messages.is_empty());
        assert!(session_manager.system_message.is_none());
    }

    #[test]
    fn test_with_system_message() {
        // Create a session manager with a system message
        let system_message = "You are a helpful assistant.";
        let session_manager =
            SessionManager::new(100).with_system_message(system_message.to_string());

        // System message should be set
        assert_eq!(
            session_manager.system_message.as_ref().unwrap().content,
            system_message
        );
    }

    #[test]
    fn test_add_messages() {
        // Create a session manager and add messages
        let mut session_manager = SessionManager::new(100);

        // Add a user message
        session_manager.add_user_message("Hello assistant".to_string());
        assert_eq!(session_manager.messages.len(), 1);
        assert_eq!(session_manager.messages[0].role, "user");

        // Add an assistant message
        session_manager.add_assistant_message("Hello user".to_string());
        assert_eq!(session_manager.messages.len(), 2);
        assert_eq!(session_manager.messages[1].role, "assistant");
    }

    #[test]
    fn test_get_messages_for_api() {
        // Create a session manager with system message and conversation
        let mut session_manager = SessionManager::new(100)
            .with_system_message("You are a helpful assistant.".to_string());

        // Add some messages
        session_manager.add_user_message("Hello assistant".to_string());
        session_manager.add_assistant_message("Hello user".to_string());

        // Get messages for API
        let api_messages = session_manager.get_messages_for_api();

        // Should include system message at the beginning
        assert_eq!(api_messages.len(), 3);
        assert_eq!(api_messages[0].role, "system");
        assert_eq!(api_messages[1].role, "user");
        assert_eq!(api_messages[2].role, "assistant");
    }

    #[test]
    fn test_clear() {
        // Create a session manager with messages
        let mut session_manager =
            SessionManager::new(100).with_system_message("System message".to_string());
        session_manager.add_user_message("User message".to_string());

        // Clear the session
        session_manager.clear();

        // Messages should be empty, but system message should remain
        assert!(session_manager.messages.is_empty());
        assert!(session_manager.system_message.is_some());
    }

    #[test]
    fn test_message_count() {
        // Create a session manager
        let mut session_manager = SessionManager::new(100);

        // Add messages
        session_manager.add_user_message("User message".to_string());
        session_manager.add_assistant_message("Assistant message".to_string());

        // Check message count
        assert_eq!(session_manager.message_count(), 2);
    }

    #[test]
    fn test_replace_with_summary() {
        // Create a session manager with some messages
        let mut session_manager = SessionManager::new(100);
        for i in 0..5 {
            session_manager.add_user_message(format!("User message {i}"));
            session_manager.add_assistant_message(format!("Assistant message {i}"));
        }

        // Record the original count
        let original_count = session_manager.messages.len();

        // Replace with a summary
        let summary = "This is a summary of the conversation.";
        session_manager.replace_with_summary(summary.to_string());

        // Should now have only one message (the summary)
        assert_eq!(session_manager.messages.len(), 1);
        assert!(session_manager.messages.len() < original_count);
        assert!(session_manager.messages[0].content.contains(summary));
        assert_eq!(session_manager.messages[0].role, "system");
    }
}
