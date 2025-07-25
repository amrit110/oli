use crate::agent::tools::{get_tool_definitions, ToolCall as AgentToolCall};
use crate::apis::api_client::{
    CompletionOptions, DynApiClient, Message, ToolCall as ApiToolCall, ToolDefinition, ToolResult,
};
use crate::prompts::add_working_directory_to_prompt;
use anyhow::{Context, Result};
use serde_json::{self, Value};
use tokio::sync::mpsc;

pub struct AgentExecutor {
    api_client: DynApiClient,
    conversation: Vec<Message>,
    tool_definitions: Vec<ToolDefinition>,
    progress_sender: Option<mpsc::Sender<String>>,
    working_directory: Option<String>,
}

impl AgentExecutor {
    pub fn new(api_client: DynApiClient) -> Self {
        let tool_defs = get_tool_definitions()
            .into_iter()
            .map(|def| ToolDefinition {
                name: def["name"].as_str().unwrap_or("").to_string(),
                description: def["description"].as_str().unwrap_or("").to_string(),
                parameters: def["parameters"].clone(),
            })
            .collect();

        Self {
            api_client,
            conversation: Vec::new(),
            tool_definitions: tool_defs,
            progress_sender: None,
            working_directory: None,
        }
    }

    pub fn set_working_directory(&mut self, working_dir: String) {
        self.working_directory = Some(working_dir.clone());

        // Update any existing system message with working directory information
        let has_system = self.conversation.iter().any(|msg| msg.role == "system");
        if has_system {
            // Find and update the system message with working directory info
            for msg in &mut self.conversation {
                if msg.role == "system" {
                    // Only add working directory if it's not already there
                    if !msg.content.contains("## WORKING DIRECTORY") {
                        // Add working directory section to end of system message
                        msg.content = add_working_directory_to_prompt(&msg.content, &working_dir);
                    }
                    break;
                }
            }
        }
    }

    pub fn set_conversation_history(&mut self, mut history: Vec<Message>) {
        // If we have a working directory, ensure any system message includes it
        if let Some(cwd) = &self.working_directory {
            for msg in &mut history {
                if msg.role == "system" && !msg.content.contains("## WORKING DIRECTORY") {
                    // Add working directory section
                    msg.content = add_working_directory_to_prompt(&msg.content, cwd);
                }
            }
        }

        self.conversation = history;
    }

    pub fn get_conversation_history(&self) -> Vec<Message> {
        self.conversation.clone()
    }

    pub fn with_progress_sender(mut self, sender: mpsc::Sender<String>) -> Self {
        self.progress_sender = Some(sender);
        self
    }

    pub fn add_system_message(&mut self, content: String) {
        // If we have a working directory, ensure it's included in the system message
        let system_content = if let Some(cwd) = &self.working_directory {
            add_working_directory_to_prompt(&content, cwd)
        } else {
            content
        };

        // Remove any existing system message to avoid duplicates
        self.conversation.retain(|msg| msg.role != "system");

        // Add the new system message
        self.conversation.push(Message::system(system_content));

        // Make sure system message is at the beginning
        self.conversation.sort_by(|a, b| {
            if a.role == "system" {
                std::cmp::Ordering::Less
            } else if b.role == "system" {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
    }

    pub fn add_user_message(&mut self, content: String) {
        self.conversation.push(Message::user(content));
    }

    pub async fn execute(&mut self) -> Result<String> {
        // Log working directory if available
        self.log_working_directory().await;
        if let Some(cwd) = &self.working_directory {
            self.add_system_message(format!("## WORKING DIRECTORY\n{cwd}"));
        }

        // Create standard completion options
        let options = self.create_completion_options();

        // Get initial completion
        let (content, tool_calls) = self.get_initial_completion(&options).await?;

        // If no tool calls, just return the response
        if tool_calls.is_none() {
            self.add_assistant_response(&content, &None);
            return Ok(content);
        }

        // Process tool calls iteratively
        let result = self
            .process_tool_calls(content, tool_calls, options)
            .await?;

        Ok(result)
    }

    // Helper method to log working directory
    async fn log_working_directory(&self) {
        if let (Some(cwd), Some(sender)) = (&self.working_directory, &self.progress_sender) {
            let _ = sender
                .send(format!("[debug] Working directory: {cwd}"))
                .await;
        }
    }

    // Helper method to create standard completion options
    fn create_completion_options(&self) -> CompletionOptions {
        CompletionOptions {
            temperature: Some(0.25),
            top_p: Some(0.95),
            max_tokens: Some(4096),
            tools: Some(self.tool_definitions.clone()),
            require_tool_use: false,
            json_schema: None,
        }
    }

    // Helper method to get initial completion
    async fn get_initial_completion(
        &self,
        options: &CompletionOptions,
    ) -> Result<(String, Option<Vec<ApiToolCall>>)> {
        self.api_client
            .complete_with_tools(self.conversation.clone(), options.clone(), None)
            .await
    }

    // Helper method to add an assistant's response to the conversation
    fn add_assistant_response(&mut self, content: &str, tool_calls: &Option<Vec<ApiToolCall>>) {
        add_assistant_message_to_conversation(&mut self.conversation, content, tool_calls);
    }

    // Process tool calls in a loop until task is complete
    async fn process_tool_calls(
        &mut self,
        initial_content: String,
        initial_tool_calls: Option<Vec<ApiToolCall>>,
        options: CompletionOptions,
    ) -> Result<String> {
        // Add the assistant's message with tool calls to the conversation
        self.add_assistant_response(&initial_content, &initial_tool_calls);

        // Process tool calls in a loop until task is complete
        let mut current_content = initial_content;
        let mut current_tool_calls = initial_tool_calls;
        let mut loop_count = 0;
        const MAX_LOOPS: usize = 100; // Limit for tool call loops
        let mut task_completed = false;

        while let Some(ref calls) = current_tool_calls {
            // Check for loop limits and log progress
            if self
                .check_loop_limits(&mut loop_count, &mut task_completed, MAX_LOOPS)
                .await
            {
                break;
            }

            // Execute all tool calls
            let tool_results = self.execute_tool_calls(calls, loop_count).await;

            // Get next completion with appropriate options
            let (next_content, next_tool_calls, is_complete) = self
                .get_next_completion(tool_results, loop_count, MAX_LOOPS, &options)
                .await?;

            // Update state for next iteration
            current_content = next_content;
            current_tool_calls = next_tool_calls;

            // Update task completion status
            if is_complete {
                task_completed = true;
            }

            // Break if task is complete or if no more tool calls
            if task_completed || current_tool_calls.is_none() {
                break;
            }

            // Log warning if approaching max loops
            self.log_approaching_max_loops(loop_count, MAX_LOOPS).await;
        }

        // Request final summary if needed
        if !task_completed && current_tool_calls.is_none() && loop_count < MAX_LOOPS - 1 {
            current_content = self.request_final_summary(&options).await?;
        }

        // Add final response to conversation
        self.add_assistant_response(&current_content, &current_tool_calls);

        Ok(current_content)
    }

    // Check loop limits and log progress
    async fn check_loop_limits(
        &self,
        loop_count: &mut usize,
        task_completed: &mut bool,
        max_loops: usize,
    ) -> bool {
        // Increment loop counter
        *loop_count += 1;

        // Safety check to prevent truly infinite loops
        if *loop_count > max_loops {
            if let Some(sender) = &self.progress_sender {
                let _ = sender
                    .send(format!(
                        "Reached maximum number of tool call loops ({max_loops}). Forcing completion."
                    ))
                    .await;
            }
            // Force task completion on max loops
            *task_completed = true;
            return true;
        }

        // Log current iteration for debugging
        if let Some(sender) = &self.progress_sender {
            let _ = sender
                .send(format!("Tool iteration {loop_count}/{max_loops}"))
                .await;
        }

        false
    }

    // Get next completion with appropriate options
    async fn get_next_completion(
        &self,
        tool_results: Vec<ToolResult>,
        loop_count: usize,
        max_loops: usize,
        base_options: &CompletionOptions,
    ) -> Result<(String, Option<Vec<ApiToolCall>>, bool)> {
        // Determine whether to request task completion
        let completion_threshold = determine_completion_threshold(loop_count);
        let should_check_completion =
            should_request_completion(loop_count, max_loops, completion_threshold);

        // Create appropriate options
        let next_options = if should_check_completion {
            self.create_completion_check_options(base_options)
        } else {
            base_options.clone()
        };

        // Request completion with tool results
        let (next_content, next_tool_calls) = self
            .api_client
            .complete_with_tools(self.conversation.clone(), next_options, Some(tool_results))
            .await?;

        // Process response to check for completion status
        let (processed_content, is_complete) = process_response(&next_content);

        Ok((processed_content, next_tool_calls, is_complete))
    }

    // Create options for checking task completion
    fn create_completion_check_options(
        &self,
        base_options: &CompletionOptions,
    ) -> CompletionOptions {
        CompletionOptions {
            require_tool_use: false,
            json_schema: Some(
                r#"{
                    "type": "object",
                    "properties": {
                        "taskComplete": {
                            "type": "boolean",
                            "description": "Whether the task is fully complete and no more tool calls are needed"
                        },
                        "finalSummary": {
                            "type": "string",
                            "description": "Final comprehensive summary of findings and results"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation of why the task is or is not complete"
                        }
                    },
                    "required": ["taskComplete", "finalSummary"]
                }"#
                .to_string(),
            ),
            ..(base_options.clone())
        }
    }

    // Log warning if approaching max loops
    async fn log_approaching_max_loops(&self, loop_count: usize, max_loops: usize) {
        if loop_count >= max_loops - 10 && loop_count % 5 == 0 {
            if let Some(sender) = &self.progress_sender {
                let _ = sender
                    .send(
                        "Approaching maximum iterations, requesting task completion check."
                            .to_string(),
                    )
                    .await;
            }
        }
    }

    // Request a final summary when no more tool calls but not explicitly completed
    async fn request_final_summary(&self, base_options: &CompletionOptions) -> Result<String> {
        if let Some(sender) = &self.progress_sender {
            let _ = sender
                .send("Task appears complete, requesting final summary.".to_string())
                .await;
        }

        let final_options = CompletionOptions {
            require_tool_use: false,
            json_schema: Some(
                r#"{
                    "type": "object",
                    "properties": {
                        "finalSummary": {
                            "type": "string",
                            "description": "Final comprehensive summary of findings and results"
                        }
                    },
                    "required": ["finalSummary"]
                }"#
                .to_string(),
            ),
            ..(base_options.clone())
        };

        // Request final summary
        let (final_content, _) = self
            .api_client
            .complete_with_tools(self.conversation.clone(), final_options, None)
            .await?;

        let (processed_content, _) = process_response(&final_content);
        Ok(processed_content)
    }

    async fn execute_tool_calls(
        &mut self,
        calls: &[ApiToolCall],
        _loop_count: usize,
    ) -> Vec<ToolResult> {
        let mut results = Vec::with_capacity(calls.len());

        for (i, call) in calls.iter().enumerate() {
            // Send tool execution progress message
            if let Some(sender) = &self.progress_sender {
                let _ = sender
                    .send(format!("⏺ [{}] Executing {}...", call.name, call.name))
                    .await;
            }

            // Parse the tool call into our enum
            let tool_call: AgentToolCall = match parse_tool_call(&call.name, &call.arguments) {
                Ok(tc) => tc,
                Err(e) => {
                    send_error_message(
                        &self.progress_sender,
                        &format!("Failed to parse tool call: {e}"),
                    )
                    .await;

                    // Add error result and continue to next tool call
                    let tool_call_id = call.id.clone().unwrap_or_else(|| format!("tool_{i}"));
                    let error_message = format!("ERROR PARSING TOOL CALL: {e}. Please check the format of your arguments and try again.");

                    self.add_tool_result_to_conversation(&tool_call_id, &error_message);
                    results.push(ToolResult {
                        tool_call_id,
                        output: error_message,
                    });

                    continue;
                }
            };

            // Execute the tool with preview for file modification tools
            let result = execute_tool_with_preview(&tool_call, call, &self.progress_sender).await;

            // Create a valid tool result ID
            let tool_call_id = call.id.clone().unwrap_or_else(|| format!("tool_{i}"));

            // Send tool execution completed message
            if let Some(sender) = &self.progress_sender {
                let _ = sender.send("[TOOL_EXECUTED]".to_string()).await;
            }

            // Add tool result to conversation and results collection
            self.add_tool_result_to_conversation(&tool_call_id, &result);
            results.push(ToolResult {
                tool_call_id,
                output: result,
            });
        }

        results
    }

    fn add_tool_result_to_conversation(&mut self, tool_call_id: &str, result: &str) {
        self.conversation.push(Message {
            role: "user".to_string(),
            content: format!("Tool result for call {tool_call_id}: {result}"),
        });
    }
}

// Helper functions to improve readability

fn add_assistant_message_to_conversation(
    conversation: &mut Vec<Message>,
    content: &str,
    tool_calls: &Option<Vec<ApiToolCall>>,
) {
    if let Some(calls) = tool_calls {
        // Create a JSON object with both content and tool calls
        let message_with_tools = serde_json::json!({
            "content": content,
            "tool_calls": calls.iter().map(|call| {
                serde_json::json!({
                    "id": call.id.clone().unwrap_or_default(),
                    "name": call.name.clone(),
                    "arguments": call.arguments.clone()
                })
            }).collect::<Vec<_>>()
        });

        // Store as JSON string in the message
        conversation.push(Message::assistant(
            serde_json::to_string(&message_with_tools).unwrap_or_else(|_| content.to_string()),
        ));
    } else {
        // No tool calls, just store the content directly
        conversation.push(Message::assistant(content.to_string()));
    }
}

/// Calculate a dynamic completion threshold based on loop count
/// As loop count increases, we become more likely to ask if the task is complete
pub fn determine_completion_threshold(loop_count: usize) -> usize {
    // Initial check after a few iterations, then gradually increase frequency
    match loop_count {
        0..=2 => 1000, // Don't check in first couple iterations
        3..=6 => 10,   // 10% chance between iterations 3-6
        7..=15 => 5,   // 20% chance between iterations 7-15
        16..=25 => 3,  // 33% chance between iterations 16-25
        26..=40 => 2,  // 50% chance between iterations 26-40
        _ => 1,        // Always check after iteration 40
    }
}

/// Determine if we should ask the LLM to check if the task is complete
pub fn should_request_completion(loop_count: usize, max_loops: usize, threshold: usize) -> bool {
    // Always check completion when approaching max loops
    if loop_count >= max_loops - 5 {
        return true;
    }

    // Periodically check based on threshold
    if threshold == 1 || loop_count % threshold == 0 {
        return true;
    }

    // Also check after specific checkpoints
    matches!(loop_count, 5 | 10 | 15 | 20 | 30 | 50 | 75)
}

/// Process the LLM response, extracting content and checking if task is complete
/// Returns (processed_content, is_complete)
pub fn process_response(content: &str) -> (String, bool) {
    if content.trim().starts_with('{') && content.trim().ends_with('}') {
        // Try to parse as JSON
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(content) {
            // Check for task completion flag
            let is_complete = json
                .get("taskComplete")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            // Extract finalSummary if available
            if let Some(summary) = json.get("finalSummary").and_then(|s| s.as_str()) {
                return (summary.to_string(), is_complete);
            }
        }
    }

    (content.to_string(), false)
}

async fn send_error_message(sender: &Option<mpsc::Sender<String>>, message: &str) {
    if let Some(sender) = sender {
        let _ = sender.send(format!("[error] {message}")).await;
    }
}

async fn execute_tool_with_preview(
    tool_call: &AgentToolCall,
    call: &ApiToolCall,
    progress_sender: &Option<mpsc::Sender<String>>,
) -> String {
    // Check if tool needs diff preview
    let needs_diff_preview = matches!(call.name.as_str(), "Edit" | "Write");

    let result = if needs_diff_preview {
        // Handle file modification tools with diff preview
        match tool_call {
            AgentToolCall::Edit(params) => {
                use crate::tools::fs::file_ops::FileOps;
                use std::path::PathBuf;

                // Generate diff without making changes
                let path = PathBuf::from(&params.file_path);
                match FileOps::generate_edit_diff(
                    &path,
                    &params.old_string,
                    &params.new_string,
                    params.expected_replacements,
                ) {
                    Ok((_, diff)) => {
                        // Send diff as progress message
                        if let Some(sender) = progress_sender {
                            let _ = sender.send(diff.clone()).await;
                        }
                        // Execute the tool
                        tool_call.execute()
                    }
                    Err(e) => Err(e),
                }
            }
            AgentToolCall::Write(params) => {
                use crate::tools::fs::file_ops::FileOps;
                use std::path::PathBuf;

                // Generate diff without making changes
                let path = PathBuf::from(&params.file_path);
                match FileOps::generate_write_diff(&path, &params.content) {
                    Ok((diff, _)) => {
                        // Send diff as progress message
                        if let Some(sender) = progress_sender {
                            let _ = sender.send(diff.clone()).await;
                        }
                        // Execute the tool
                        tool_call.execute()
                    }
                    Err(e) => Err(e),
                }
            }
            _ => tool_call.execute(), // Shouldn't happen, but fallback
        }
    } else {
        // For non-file operations, execute normally
        tool_call.execute()
    };

    match result {
        Ok(output) => output,
        Err(e) => format!("ERROR EXECUTING TOOL: {e}"),
    }
}

fn parse_tool_call(name: &str, args: &Value) -> Result<AgentToolCall> {
    match name {
        "Read" => {
            let params =
                serde_json::from_value(args.clone()).context("Failed to parse Read parameters")?;
            Ok(AgentToolCall::Read(params))
        }
        "Glob" => {
            let params =
                serde_json::from_value(args.clone()).context("Failed to parse Glob parameters")?;
            Ok(AgentToolCall::Glob(params))
        }
        "Grep" => {
            let params =
                serde_json::from_value(args.clone()).context("Failed to parse Grep parameters")?;
            Ok(AgentToolCall::Grep(params))
        }
        "LS" => {
            let params =
                serde_json::from_value(args.clone()).context("Failed to parse LS parameters")?;
            Ok(AgentToolCall::LS(params))
        }
        "Edit" => {
            let params =
                serde_json::from_value(args.clone()).context("Failed to parse Edit parameters")?;
            Ok(AgentToolCall::Edit(params))
        }
        "Write" => {
            let params =
                serde_json::from_value(args.clone()).context("Failed to parse Write parameters")?;
            Ok(AgentToolCall::Write(params))
        }
        "Bash" => {
            let params =
                serde_json::from_value(args.clone()).context("Failed to parse Bash parameters")?;
            Ok(AgentToolCall::Bash(params))
        }
        _ => Err(anyhow::anyhow!("Unknown tool: {}", name)),
    }
}
