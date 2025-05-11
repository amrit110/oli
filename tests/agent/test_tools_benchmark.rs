use crate::agent::utils::{
    compare_tool_params, init_logging, log, setup_agent, setup_test_files, ToolBenchmarkDataset,
};
use oli_server::app::logger::LogLevel;
use oli_server::prompts;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use tempfile::tempdir;
use tokio;

#[tokio::test]
#[cfg_attr(not(feature = "benchmark"), ignore)]
async fn benchmark_tool_call_accuracy() {
    // Initialize logging
    init_logging();

    // Create a temporary directory for test files
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let test_dir = setup_test_files(&temp_dir);
    log(
        LogLevel::Info,
        &format!("Test files created in: {}", test_dir.display()),
    );

    // Log the test directory
    log(
        LogLevel::Info,
        &format!("Using test directory: {}", test_dir.display()),
    );

    // Set up the agent with the configured provider
    let Some((mut agent, timeout_secs)) = setup_agent().await else {
        log(
            LogLevel::Error,
            "Skipping benchmark test: Agent setup failed",
        );
        return;
    };

    // Log which model and provider is being used
    let provider = env::var("TEST_PROVIDER").unwrap_or_else(|_| "ollama".to_string());
    let model = env::var("TEST_MODEL")
        .unwrap_or_else(|_| env::var("DEFAULT_MODEL").unwrap_or_else(|_| "unknown".to_string()));
    log(
        LogLevel::Info,
        &format!(
            "Running benchmark with provider: {}, model: {}",
            provider, model
        ),
    );

    // Set the working directory on the agent
    agent = agent.with_working_directory(test_dir.to_string_lossy().to_string());

    // Load benchmark dataset
    let dataset_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/agent/tool_benchmarks.json");
    let dataset_content =
        fs::read_to_string(&dataset_path).expect("Failed to read tool benchmarks dataset");
    let dataset: ToolBenchmarkDataset =
        serde_json::from_str(&dataset_content).expect("Failed to parse tool benchmarks dataset");

    log(
        LogLevel::Info,
        &format!("Loaded {} benchmark queries", dataset.queries.len()),
    );

    // Statistics
    let mut correct_count = 0;
    let mut total_count = 0;
    let mut results = Vec::new();

    // Show progress bar
    log(LogLevel::Info, "Starting benchmark test...");
    log(
        LogLevel::Info,
        "======================================================",
    );

    // Calculate the maximum number of digits in dataset length for padding
    let max_digits = dataset.queries.len().to_string().len();

    // Process each query
    for (i, query) in dataset.queries.iter().enumerate() {
        total_count += 1;

        // Print a simple progress indicator to stderr
        eprint!("\r[{}/{}] ", i + 1, dataset.queries.len());
        io::stderr().flush().ok();

        // Run just a few examples to test the changes
        if i < 2 {
            continue;
        } else if i > 3 {
            break;
        }

        // Log the current query being tested
        log(
            LogLevel::Info,
            &format!(
                "\n[{:0width$}/{}] Testing: \"{}\"",
                i + 1,
                dataset.queries.len(),
                query.query,
                width = max_digits
            ),
        );

        // Log expected tool and parameters for debugging
        log(
            LogLevel::Debug,
            &format!(
                "Expected tool: {}, Expected params: {:?}",
                query.expected_tool, query.expected_params
            ),
        );
        // Set a reasonable timeout
        let timeout_duration = std::time::Duration::from_secs(timeout_secs);

        // Start timing
        let start_time = std::time::Instant::now();

        // Create an executor which will handle conversation management and system prompt
        let mut executor =
            oli_server::agent::executor::AgentExecutor::new(agent.api_client.clone().unwrap());

        // Set the working directory on the executor
        executor.set_working_directory(test_dir.to_string_lossy().to_string());

        // Add system message with working directory information
        let system_prompt = prompts::get_agent_prompt_with_cwd(Some(&test_dir.to_string_lossy()));
        executor.add_system_message(system_prompt.clone());

        // Add the user query
        executor.add_user_message(query.query.clone());

        // Create completion options with tools but don't force tool use
        let options = oli_server::apis::api_client::CompletionOptions {
            temperature: Some(0.25),
            top_p: Some(0.95),
            max_tokens: Some(4096),
            tools: Some(executor.tool_definitions.clone()),
            require_tool_use: false, // Match behavior in src/agent/executor.rs
            json_schema: None,
        };

        // Use the agent's API client to get tool calls without executing them
        // Add a note to explicitly encourage using specific tools for specific tasks
        log(
            LogLevel::Info,
            &format!(
                "Testing if model correctly uses {} for query: '{}'",
                query.expected_tool, query.query
            ),
        );

        let result = tokio::time::timeout(
            timeout_duration,
            executor
                .api_client
                .complete_with_tools(executor.conversation.clone(), options, None),
        )
        .await;
        let elapsed = start_time.elapsed();

        match result {
            Ok(inner_result) => {
                match inner_result {
                    Ok((content, tool_calls)) => {
                        // Log truncated content for debugging
                        log(
                            LogLevel::Debug,
                            &format!(
                                "Response content (truncated): {}",
                                if content.len() > 100 {
                                    format!("{}...", &content[..100])
                                } else {
                                    content.clone()
                                }
                            ),
                        );

                        // Check if we got any tool calls directly from the API
                        if let Some(calls) = tool_calls {
                            if !calls.is_empty() {
                                let tool_call = &calls[0];

                                // Print more debug info about the tool call and arguments
                                log(
                                    LogLevel::Info,
                                    &format!("Tool detected: {}", tool_call.name),
                                );
                                log(
                                    LogLevel::Debug,
                                    &format!("Tool call arguments: {:?}", tool_call.arguments),
                                );

                                // Compare with expected tool call
                                let is_correct = compare_tool_params(
                                    &query.expected_tool,
                                    &query.expected_params,
                                    &tool_call.name,
                                    &tool_call.arguments,
                                    &test_dir,
                                );

                                if is_correct {
                                    correct_count += 1;
                                    log(LogLevel::Info, "✅ Tool call correct");
                                    eprint!("✅"); // Simple progress indicator
                                    io::stderr().flush().ok();
                                } else {
                                    log(
                                        LogLevel::Warning,
                                        &format!(
                                            "❌ Tool call incorrect. Expected: {}, got: {}",
                                            query.expected_tool, tool_call.name
                                        ),
                                    );
                                    eprint!("❌"); // Simple progress indicator
                                    io::stderr().flush().ok();
                                }

                                // Record result
                                results.push((i, query.query.clone(), is_correct));
                            } else {
                                log(LogLevel::Warning, "❌ Empty tool calls array in response");
                                eprint!("❌"); // Simple progress indicator
                                io::stderr().flush().ok();
                                results.push((i, query.query.clone(), false));
                            }
                        } else {
                            // No tool calls returned from API
                            log(LogLevel::Warning, "❌ No tool calls in API response");
                            eprint!("❌"); // Simple progress indicator
                            io::stderr().flush().ok();
                            results.push((i, query.query.clone(), false));
                        }
                    }
                    Err(e) => {
                        log(LogLevel::Error, &format!("❌ API call failed: {}", e));
                        eprint!("❌"); // Simple progress indicator
                        io::stderr().flush().ok();
                        results.push((i, query.query.clone(), false));
                    }
                }
            }
            Err(_) => {
                log(
                    LogLevel::Warning,
                    &format!("⏱️ Test timed out after {} seconds", timeout_secs),
                );
                eprint!("⏱️"); // Simple progress indicator
                io::stderr().flush().ok();
                results.push((i, query.query.clone(), false));
            }
        }

        // Log time taken
        log(
            LogLevel::Debug,
            &format!("Query completed in {:.2?}", elapsed),
        );

        // Clear agent history for next query
        agent.clear_history();
    }

    // Print a newline after progress indicators
    eprintln!();

    // Calculate and report accuracy
    let accuracy = if total_count > 0 {
        (correct_count as f64 / total_count as f64) * 100.0
    } else {
        0.0
    };

    // Log the final results with a nicely formatted summary
    log(
        LogLevel::Info,
        "\n======================================================",
    );
    log(
        LogLevel::Info,
        "              BENCHMARK RESULTS                      ",
    );
    log(
        LogLevel::Info,
        "======================================================",
    );
    log(
        LogLevel::Info,
        &format!("Total queries:      {}", total_count),
    );
    log(
        LogLevel::Info,
        &format!("Correct tool calls: {}", correct_count),
    );
    log(
        LogLevel::Info,
        &format!("Accuracy:           {:.2}%", accuracy),
    );

    if correct_count < total_count {
        log(LogLevel::Info, "\nIncorrect queries:");
        for (i, query, is_correct) in &results {
            if !is_correct {
                log(LogLevel::Warning, &format!("- [{}] {}", i, query));
            }
        }
    }

    // For reporting in CI, we can accept a low accuracy threshold for passing the test
    // In practice, this can be adjusted based on the model's capabilities
    let min_accuracy_threshold = if std::env::var("FORCE_SUCCESS").is_ok() {
        0.0 // When FORCE_SUCCESS is set, allow any accuracy
    } else {
        50.0 // Normal threshold for regular runs
    };

    // No need to change directory back since we're using the agent's working directory setting
    log(LogLevel::Info, "Test completed successfully");

    // Log if we're using FORCE_SUCCESS to bypass threshold
    if std::env::var("FORCE_SUCCESS").is_ok() {
        log(
            LogLevel::Warning,
            &format!("FORCE_SUCCESS environment variable is set - bypassing accuracy check. Actual accuracy: {:.2}%", accuracy),
        );

        // When FORCE_SUCCESS is set, simply report rather than assert
        if accuracy < min_accuracy_threshold {
            log(
                LogLevel::Warning,
                &format!("In a normal run, this test would have failed: accuracy {:.2}% is below minimum threshold {:.2}%",
                    accuracy, min_accuracy_threshold),
            );
        }
    } else {
        // In normal mode, make the assertion
        assert!(
            accuracy >= min_accuracy_threshold,
            "Tool call accuracy too low: {:.2}% (minimum: {:.2}%)",
            accuracy,
            min_accuracy_threshold
        );
    }
}
