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
async fn benchmark_tool_call_correctness_and_efficiency() {
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
    let mut total_tool_calls = 0;
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
        // if i < 0 {
        //     continue;
        // } else if i > 1 {
        //     break;
        // }

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
            require_tool_use: false,
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
                                // Track the total number of tool calls for efficiency metric
                                total_tool_calls += calls.len();

                                // Track whether any tool call was correct
                                let mut found_correct_tool = false;

                                // Log number of tool calls made for this query
                                log(
                                    LogLevel::Info,
                                    &format!("Number of tool calls: {}", calls.len()),
                                );

                                // Check each tool call to see if any matches the expected one
                                for tool_call in &calls {
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
                                        found_correct_tool = true;
                                        log(LogLevel::Info, "✅ Found correct tool call");
                                        break; // Found a correct tool call, no need to check others
                                    }
                                }

                                // Update correctness metric
                                if found_correct_tool {
                                    correct_count += 1;
                                    log(LogLevel::Info, "✅ Correctness: Tool call found");
                                    eprint!("✅"); // Simple progress indicator
                                    io::stderr().flush().ok();
                                } else {
                                    log(
                                        LogLevel::Warning,
                                        &format!(
                                            "❌ Correctness: Tool call incorrect. Expected: {}, not found in {} calls",
                                            query.expected_tool, calls.len()
                                        ),
                                    );
                                    eprint!("❌"); // Simple progress indicator
                                    io::stderr().flush().ok();
                                }

                                // Record result
                                results.push((
                                    i,
                                    query.query.clone(),
                                    found_correct_tool,
                                    calls.len(),
                                ));
                            } else {
                                log(LogLevel::Warning, "❌ Empty tool calls array in response");
                                eprint!("❌"); // Simple progress indicator
                                io::stderr().flush().ok();
                                results.push((i, query.query.clone(), false, 0));
                            }
                        } else {
                            // No tool calls returned from API
                            log(LogLevel::Warning, "❌ No tool calls in API response");
                            eprint!("❌"); // Simple progress indicator
                            io::stderr().flush().ok();
                            results.push((i, query.query.clone(), false, 0));
                        }
                    }
                    Err(e) => {
                        log(LogLevel::Error, &format!("❌ API call failed: {}", e));
                        eprint!("❌"); // Simple progress indicator
                        io::stderr().flush().ok();
                        results.push((i, query.query.clone(), false, 0));
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
                results.push((i, query.query.clone(), false, 0));
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

    // Calculate correctness and efficiency metrics
    let correctness = if total_count > 0 {
        (correct_count as f64 / total_count as f64) * 100.0
    } else {
        0.0
    };

    // Efficiency: Ideally each query should have exactly 1 tool call
    // Lower values mean the model made unnecessary extra calls
    let efficiency = if total_count > 0 {
        if total_tool_calls >= total_count {
            (total_count as f64 / total_tool_calls as f64) * 100.0
        } else {
            // If we got fewer tool calls than queries, this means some queries had no tools
            // called at all - which is a failure case we should count against efficiency
            (total_tool_calls as f64 / total_count as f64) * 100.0
        }
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
        &format!("Total tool calls:   {}", total_tool_calls),
    );
    log(
        LogLevel::Info,
        &format!("Correct queries:    {}", correct_count),
    );
    log(
        LogLevel::Info,
        &format!("Correctness:        {:.2}%", correctness),
    );
    log(
        LogLevel::Info,
        &format!("Efficiency:         {:.2}%", efficiency),
    );
    log(
        LogLevel::Info,
        &format!("Ideal tool calls:   {} (1 per query)", total_count),
    );

    // Print queries with incorrect tool calls or excessive tool use
    if correct_count < total_count || total_tool_calls > total_count {
        log(LogLevel::Info, "\nIncorrect or inefficient queries:");
        for (i, query, is_correct, num_calls) in &results {
            if !is_correct {
                log(
                    LogLevel::Warning,
                    &format!("- [{}] Incorrect: {}", i, query),
                );
            } else if *num_calls > 1 {
                log(
                    LogLevel::Warning,
                    &format!("- [{}] Inefficient ({} calls): {}", i, num_calls, query),
                );
            }
        }
    }

    // For reporting in CI, we can accept a low threshold for passing the test
    let min_correctness_threshold = if std::env::var("FORCE_SUCCESS").is_ok() {
        0.0 // When FORCE_SUCCESS is set, allow any correctness
    } else {
        50.0 // Normal threshold for regular runs
    };

    // No need to change directory back since we're using the agent's working directory setting
    log(LogLevel::Info, "Test completed successfully");

    // Log if we're using FORCE_SUCCESS to bypass threshold
    if std::env::var("FORCE_SUCCESS").is_ok() {
        log(
            LogLevel::Warning,
            &format!(
                "FORCE_SUCCESS environment variable is set - bypassing correctness check. Actual correctness: {:.2}%, efficiency: {:.2}%",
                correctness, efficiency
            ),
        );

        // When FORCE_SUCCESS is set, simply report rather than assert
        if correctness < min_correctness_threshold {
            log(
                LogLevel::Warning,
                &format!("In a normal run, this test would have failed: correctness {:.2}% is below minimum threshold {:.2}%",
                    correctness, min_correctness_threshold),
            );
        }
    } else {
        // In normal mode, make the assertion
        assert!(
            correctness >= min_correctness_threshold,
            "Tool call correctness too low: {:.2}% (minimum: {:.2}%)",
            correctness,
            min_correctness_threshold
        );
    }
}
