use oli_server::apis::api_client::SessionManager;
use oli_server::app::history::{ContextCompressor, ConversationSummary};
use oli_server::{Agent, App, AppState, LLMProvider};

#[test]
fn test_conversation_char_count() {
    let mut app = App::new();
    app.state = AppState::Chat;

    // Empty conversation should have 0 chars
    assert_eq!(app.conversation_char_count(), 0);

    // Add some messages
    app.messages.push("Hello".to_string());
    app.messages.push("World".to_string());

    // Should count "Hello" (5) + "World" (5) = 10
    assert_eq!(app.conversation_char_count(), 10);
}

#[test]
fn test_should_compress() {
    let mut app = App::new();
    app.state = AppState::Chat;

    // Empty conversation should not need summarization
    assert!(!app.should_compress());

    // Add enough messages to trigger summarization by count (1000 is the threshold in history.rs)
    for i in 0..1001 {
        app.messages.push(format!("Message {i}"));
    }

    // Should now need summarization
    assert!(app.should_compress());

    // Reset and try with character count
    app.messages.clear();

    // Add a single large message (1,000,000 chars is the threshold in history.rs)
    app.messages.push("A".repeat(1000001));

    // Should need summarization due to character count
    assert!(app.should_compress());
}

#[test]
fn test_summary_count() {
    let app = App::new();

    // Should start with 0 summaries
    assert_eq!(app.summary_count(), 0);
}

#[test]
fn test_clear_history() {
    let mut app = App::new();

    // Add some messages and summaries
    app.messages.push("Test message".to_string());

    // Add a fake summary
    app.conversation_summaries
        .push(ConversationSummary::new("Test summary".to_string(), 5, 100));

    // Clear history
    app.clear_history();

    // Summaries should be empty
    // Messages won't be empty because clear_history adds a notification message
    assert_eq!(app.messages.len(), 1);
    assert!(app.messages[0].contains("Chat history cleared"));
    assert!(app.conversation_summaries.is_empty());
}

#[test]
fn test_session_manager_integration() {
    let mut app = App::new();
    app.state = AppState::Chat;

    // Create a session manager with a system prompt
    let mut session_manager =
        SessionManager::new(100).with_system_message("Test system prompt".to_string());

    // Add some conversation messages
    session_manager.add_user_message("Hello from user".to_string());
    session_manager.add_assistant_message("Hello from assistant".to_string());

    // Set the session manager in the app
    app.session_manager = Some(session_manager);

    // Verify that the session manager contains the expected messages
    let messages = app.session_manager.as_ref().unwrap().get_messages_for_api();

    // Should have 3 messages (system + user + assistant)
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].role, "system");
    assert_eq!(messages[0].content, "Test system prompt");
    assert_eq!(messages[1].role, "user");
    assert_eq!(messages[1].content, "Hello from user");
    assert_eq!(messages[2].role, "assistant");
    assert_eq!(messages[2].content, "Hello from assistant");

    // Clearing history should also clear session manager
    app.clear_history();

    // Session manager should now be empty (except for system message)
    let messages_after_clear = app.session_manager.as_ref().unwrap().get_messages_for_api();
    assert_eq!(messages_after_clear.len(), 1); // Just the system message remains
    assert_eq!(messages_after_clear[0].role, "system");
}

#[test]
fn test_display_session_message_conversion() {
    let app = App::new();

    // Create display messages
    let display_messages = vec![
        "[user] Hello there".to_string(),
        "[assistant] Hi, how can I help?".to_string(),
        "[system] Special instruction".to_string(),
    ];

    // Convert display to session messages
    let session_messages = app.display_to_session_messages(&display_messages);

    // Verify conversion
    assert_eq!(session_messages.len(), 3);
    assert_eq!(session_messages[0].role, "user");
    assert_eq!(session_messages[0].content, "Hello there");
    assert_eq!(session_messages[1].role, "assistant");
    assert_eq!(session_messages[1].content, "Hi, how can I help?");
    assert_eq!(session_messages[2].role, "system");
    assert_eq!(session_messages[2].content, "Special instruction");

    // Convert back to display messages
    let reconverted = app.session_to_display_messages(&session_messages);

    // Verify round-trip conversion
    assert_eq!(reconverted.len(), 3);
    assert_eq!(reconverted[0], "[user] Hello there");
    assert_eq!(reconverted[1], "[assistant] Hi, how can I help?");
    assert_eq!(reconverted[2], "[system] Special instruction");
}

/// Test that verifies the agent correctly receives conversation history
/// This test mocks parts of the run implementation to verify
/// that the conversation history from the session manager is properly
/// passed to the agent.
#[test]
fn test_agent_conversation_history_integration() {
    // Create a mock agent to capture the messages
    let mut agent = Agent::new(LLMProvider::Anthropic);

    // Set up an app with conversation history
    let mut app = App::new();
    app.state = AppState::Chat;
    app.use_agent = true; // Enable agent mode

    // Create a session manager with conversation history
    let mut session =
        SessionManager::new(100).with_system_message("Test system prompt".to_string());

    // Add a conversation with multiple messages
    session.add_user_message("First user message".to_string());
    session.add_assistant_message("First assistant response".to_string());
    session.add_user_message("Second user message".to_string());
    session.add_assistant_message("Second assistant response".to_string());

    // Set the session in the app
    app.session_manager = Some(session);

    // Get messages from session manager (simulate what happens in run)
    let session_messages = app.session_manager.as_ref().unwrap().get_messages_for_api();

    // Should have 5 messages (system + 2 user + 2 assistant)
    assert_eq!(session_messages.len(), 5);

    // Add the messages to the agent (what happens in our fixed code)
    for message in session_messages {
        agent.add_message(message);
    }

    // Verify that conversation history was added to the agent
    assert_eq!(agent.get_conversation_history_for_test().len(), 5);

    // Check the first message is the system message
    let agent_messages = agent.get_conversation_history_for_test();
    assert_eq!(agent_messages[0].role, "system");
    assert_eq!(agent_messages[0].content, "Test system prompt");

    // Check the user and assistant messages are in the correct order
    assert_eq!(agent_messages[1].role, "user");
    assert_eq!(agent_messages[1].content, "First user message");
    assert_eq!(agent_messages[2].role, "assistant");
    assert_eq!(agent_messages[2].content, "First assistant response");
    assert_eq!(agent_messages[3].role, "user");
    assert_eq!(agent_messages[3].content, "Second user message");
    assert_eq!(agent_messages[4].role, "assistant");
    assert_eq!(agent_messages[4].content, "Second assistant response");
}
