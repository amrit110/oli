use anyhow::Result;
use oli_server::apis::api_client::SessionManager;
use oli_server::app::core::{App, TaskStatus, ToolExecutionStatus};
use oli_server::models::ModelConfig;
use std::{collections::HashMap, env};

// Test helpers
fn setup_app() -> Result<App> {
    // Create a new App instance
    let mut app = App::new();

    // Create a simple mock tokio runtime for the app
    app.tokio_runtime = Some(
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?,
    );

    // Ensure we have a session manager
    if app.session_manager.is_none() {
        app.session_manager = Some(SessionManager::new(100));
    }

    Ok(app)
}

#[test]
fn test_local_model_no_api_key_required() -> Result<()> {
    // Create a new App instance
    let mut app = setup_app()?;

    // Add a mock local model to the available models
    app.available_models = vec![ModelConfig {
        name: "Test Local Model (local)".into(),
        file_name: "test-model".into(),
        description: "Test local model via Ollama".into(),
        recommended_for: "Testing".into(),
        supports_agent: true,
    }];

    // Ensure no API keys are set in the environment
    env::remove_var("ANTHROPIC_API_KEY");
    env::remove_var("OPENAI_API_KEY");
    env::remove_var("GEMINI_API_KEY");

    // The test passes if this doesn't panic with an API key error
    let result = app.run("test prompt", Some(0));

    // The test should fail for other reasons (like Ollama not running)
    // but not because of missing API keys
    assert!(
        result.is_err(),
        "Expected query to fail for other reasons, but not due to API key issues"
    );

    // Verify the error is not about missing API keys
    let err_msg = result.unwrap_err().to_string();
    assert!(
        !err_msg.contains("No API key available"),
        "Test failed with API key error: {err_msg}"
    );

    Ok(())
}

#[test]
fn test_cloud_model_requires_api_key() -> Result<()> {
    // Create a new App instance
    let mut app = setup_app()?;

    // Add a mock Claude model to the available models
    app.available_models = vec![ModelConfig {
        name: "Test Claude Model".into(),
        file_name: "claude-test".into(),
        description: "Test Claude model".into(),
        recommended_for: "Testing".into(),
        supports_agent: true,
    }];

    // Ensure no API keys are set in the environment
    env::remove_var("ANTHROPIC_API_KEY");
    env::remove_var("OPENAI_API_KEY");
    env::remove_var("GEMINI_API_KEY");

    // Try to run the model, which should fail due to missing API key
    let result = app.run("test prompt", Some(0));

    // Verify the error is about missing API keys
    assert!(
        result.is_err(),
        "Expected query to fail due to missing API key"
    );

    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("No API key available"),
        "Error message should mention missing API key: {err_msg}"
    );

    Ok(())
}

#[test]
fn test_get_api_source() -> Result<()> {
    // Helper method to test API source determination
    let test_cases = vec![
        ("claude-3-opus", "Anthropic"),
        ("gpt-4", "OpenAI"),
        ("gemini-pro", "Google"),
        ("llama2 (local)", "Local"),
        ("unknown-model", "Unknown"),
    ];

    for (model, expected) in test_cases {
        let source = App::get_api_source(model);
        assert_eq!(
            source, expected,
            "API source for '{model}' should be '{expected}' but got '{source}'"
        );
    }

    Ok(())
}

#[test]
fn test_estimate_tokens() -> Result<()> {
    // Test token estimation function with different text lengths
    let test_cases = vec![
        ("", 0),                                                      // Empty text
        ("Hello", 2),                                                 // Short text
        ("This is a longer text that should be about 13 tokens", 13), // Medium text
    ];

    for (text, expected) in test_cases {
        let token_count = App::estimate_tokens(text);
        assert_eq!(
            token_count, expected,
            "Token count for '{text}' should be '{expected}' but got '{token_count}'"
        );
    }

    Ok(())
}

#[test]
fn test_validate_api_key() -> Result<()> {
    // Test API key validation

    // Local model should work with empty API key
    let result = App::validate_api_key("local-model (local)", "");
    assert!(result.is_ok(), "Local model should not require API key");

    // Cloud model should fail with empty API key
    let result = App::validate_api_key("claude-3", "");
    assert!(result.is_err(), "Cloud model should require API key");

    // Cloud model should work with valid API key
    let result = App::validate_api_key("claude-3", "test-api-key");
    assert!(result.is_ok(), "Cloud model with API key should validate");

    Ok(())
}

#[test]
fn test_extract_tool_metadata() -> Result<()> {
    // Test metadata extraction from tool messages

    // Test file path extraction
    let test_cases = vec![
        (
            "Processing file_path: \"/path/to/file.rs\"",
            Some("/path/to/file.rs".to_string()),
            None,
        ),
        (
            "Read 50 lines from file_path: /another/path.rs",
            Some("/another/path.rs".to_string()),
            Some(50),
        ),
        ("Found 25 lines matching pattern", None, Some(25)),
        ("No metadata here", None, None),
    ];

    for (message, expected_path, expected_lines) in test_cases {
        let (path, lines) = App::extract_tool_metadata(message);
        assert_eq!(
            path, expected_path,
            "File path extraction failed for '{message}'. Expected '{expected_path:?}' but got '{path:?}'"
        );
        assert_eq!(
            lines, expected_lines,
            "Line count extraction failed for '{message}'. Expected '{expected_lines:?}' but got '{lines:?}'"
        );
    }

    Ok(())
}

#[test]
fn test_get_tool_description() -> Result<()> {
    // Test tool description generation

    let file_path = "/path/to/file.rs".to_string();
    let line_count = 50;

    let test_cases = vec![
        (
            "View",
            Some(file_path.clone()),
            Some(line_count),
            "Read 50 lines (ctrl+r to expand)",
        ),
        (
            "View",
            Some(file_path.clone()),
            None,
            "Reading file contents (ctrl+r to expand)",
        ),
        ("View", None, None, "Reading file"),
        ("Glob", None, None, "Finding files by pattern"),
        ("Grep", None, None, "Searching code for pattern"),
        ("LS", None, None, "Listing directory contents"),
        ("Edit", None, None, "Modifying file"),
        ("Replace", None, None, "Replacing file contents"),
        ("Bash", None, None, "Executing command"),
        ("Unknown", None, None, "Executing tool"),
    ];

    for (tool, path, lines, expected) in test_cases {
        let description = App::get_tool_description(tool, &path, lines);
        assert_eq!(
            description, expected,
            "Description for tool '{tool}' should be '{expected}' but got '{description}'"
        );
    }

    Ok(())
}

#[test]
fn test_task_management() -> Result<()> {
    // Test task creation and management
    let mut app = setup_app()?;

    // Test task creation
    let task_id = app.create_task("Test task");
    assert!(
        app.current_task_id.is_some(),
        "Current task ID should be set after create_task"
    );
    assert_eq!(
        app.current_task_id.as_ref().unwrap(),
        &task_id,
        "Current task ID should match created task ID"
    );

    // Test tool use tracking
    app.add_tool_use();
    let task = app.current_task().unwrap();
    assert_eq!(
        task.tool_count, 1,
        "Tool count should be 1 after add_tool_use"
    );

    // Test input token tracking
    app.add_input_tokens(100);
    let task = app.current_task().unwrap();
    assert_eq!(task.input_tokens, 100, "Input token count should be 100");

    // Test task completion
    app.complete_current_task(200);
    assert!(
        app.current_task_id.is_none(),
        "Current task ID should be None after task completion"
    );

    // Get the completed task
    let task = app.tasks.iter().find(|t| t.id == task_id).unwrap();
    assert!(
        matches!(task.status, TaskStatus::Completed { .. }),
        "Task status should be Completed"
    );

    if let TaskStatus::Completed {
        output_tokens,
        tool_uses,
        ..
    } = task.status
    {
        assert_eq!(output_tokens, 200, "Output tokens should be 200");
        assert_eq!(tool_uses, 1, "Tool uses should be 1");
    }

    Ok(())
}

#[test]
fn test_task_failure() -> Result<()> {
    // Test task failure handling
    let mut app = setup_app()?;

    // Create a task
    let task_id = app.create_task("Failing task");

    // Fail the task
    app.fail_current_task("Test error");
    assert!(
        app.current_task_id.is_none(),
        "Current task ID should be None after task failure"
    );

    // Get the failed task
    let task = app.tasks.iter().find(|t| t.id == task_id).unwrap();
    assert!(
        matches!(task.status, TaskStatus::Failed(_)),
        "Task status should be Failed"
    );

    if let TaskStatus::Failed(error_message) = &task.status {
        assert_eq!(error_message, "Test error", "Error message should match");
    }

    Ok(())
}

#[test]
fn test_tool_execution_tracking() -> Result<()> {
    // Test tool execution tracking
    let mut app = setup_app()?;

    // Create a task
    let _task_id = app.create_task("Tool execution task");

    // Start a tool execution
    let tool_id = app.start_tool_execution("Test Tool").unwrap();
    assert!(
        app.tool_executions.contains_key(&tool_id),
        "Tool execution should be tracked"
    );

    let tool = app.tool_executions.get(&tool_id).unwrap();
    assert_eq!(tool.name, "Test Tool", "Tool name should match");
    assert_eq!(
        tool.status,
        ToolExecutionStatus::Running,
        "Tool should be in running state"
    );

    // Update tool progress
    let mut metadata = HashMap::new();
    metadata.insert("test_key".to_string(), serde_json::json!("test_value"));
    app.update_tool_progress(&tool_id, "Progress update", Some(metadata));

    let tool = app.tool_executions.get(&tool_id).unwrap();
    assert_eq!(
        tool.message, "Progress update",
        "Tool message should be updated"
    );
    assert!(
        tool.metadata.contains_key("test_key"),
        "Tool metadata should be updated"
    );

    // Complete the tool execution
    app.complete_tool_execution(&tool_id, "Completed", None);
    let tool = app.tool_executions.get(&tool_id).unwrap();
    assert_eq!(
        tool.status,
        ToolExecutionStatus::Success,
        "Tool should be in success state"
    );
    assert!(tool.end_time.is_some(), "Tool end time should be set");

    Ok(())
}
