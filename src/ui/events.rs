#![allow(clippy::needless_borrow)]

use crate::app::agent::AgentManager;
use crate::app::commands::CommandHandler;
use crate::app::models::ModelManager;
use crate::app::permissions::PermissionHandler;
use crate::app::state::{App, AppState};
use crate::app::utils::Scrollable;
use crate::ui::draw::ui;
use crate::ui::guards::TerminalGuard;
use crate::ui::messages::{initialize_setup_messages, process_message};
use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyModifiers};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::{io, sync::mpsc, time::Duration};
use tui_textarea::{Input, Key};

/// Main application run loop
pub fn run_app() -> Result<()> {
    // Initialize terminal
    let _guard = TerminalGuard::new()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    // Initialize application state
    let mut app = App::new();

    // Set up welcome messages
    initialize_setup_messages(&mut app);
    app.messages
        .push("DEBUG: Application started. Press Enter to begin setup.".into());

    // Create channel for events
    let (tx, rx) = mpsc::channel::<String>();

    // Initial UI draw
    terminal.draw(|f| ui(f, &mut app))?;

    // Track when we last redrew the screen to control framerate
    let mut last_redraw = std::time::Instant::now();
    let min_redraw_interval = std::time::Duration::from_millis(100);

    // Main event loop
    while app.state != AppState::Error("quit".into()) {
        // Process messages without forcing screen redraws
        process_channel_messages(&mut app, &rx, &mut terminal)?;
        process_agent_messages(&mut app, &mut terminal)?;
        process_auto_scroll(&mut app, &mut terminal)?;

        // Determine if we need to redraw based on application state
        let need_animation = app.agent_progress_rx.is_some()
            || app.permission_required
            || app.tool_execution_in_progress;

        // Throttle redraws to prevent flickering and allow scrolling to work
        let should_redraw = need_animation && last_redraw.elapsed() >= min_redraw_interval;

        // Only redraw at controlled intervals when animations are needed
        if should_redraw {
            terminal.draw(|f| ui(f, &mut app))?;
            last_redraw = std::time::Instant::now();
        }

        // Check for command mode before handling events
        if let AppState::Chat = app.state {
            if app.input.starts_with('/') {
                app.check_command_mode();
            }
        }

        // Process user input with short timeout to keep processing messages
        // This shorter poll timeout makes the UI more responsive during tool execution
        if crossterm::event::poll(Duration::from_millis(25))? {
            if let Event::Key(key) = crossterm::event::read()? {
                // Pass both the key code and the modifiers to the process_key_event function
                process_key_event(&mut app, key.code, key.modifiers, &tx, &mut terminal)?;
            }
        } else {
            // Use a very short sleep to keep checking messages frequently
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    Ok(())
}

/// Process messages from the message channel without forcing redraws
fn process_channel_messages(
    app: &mut App,
    rx: &mpsc::Receiver<String>,
    _terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    let mut received_message = false;

    while let Ok(msg) = rx.try_recv() {
        received_message = true;

        if app.debug_messages {
            app.messages
                .push(format!("DEBUG: Received message: {}", msg));
        }
        process_message(app, &msg)?;
        // Don't redraw here - let the main loop control the redraw timing
    }

    // If we received any messages, add more auto-scroll markers to ensure visibility
    if received_message {
        // Add multiple markers to ensure enough scrolling
        for _ in 0..3 {
            app.messages.push("_AUTO_SCROLL_".into());
        }
    }

    Ok(())
}

/// Process messages from the agent progress channel without forcing redraws
fn process_agent_messages(
    app: &mut App,
    _terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    // Collect messages first to avoid borrow checker issues
    let mut messages_to_process = Vec::new();
    let mut any_tool_executed = false;

    // Check for agent progress messages and collect them
    if let Some(ref agent_rx) = &app.agent_progress_rx {
        // Drain all available messages into our collection
        while let Ok(msg) = agent_rx.try_recv() {
            if msg == "[TOOL_EXECUTED]" {
                any_tool_executed = true;
            } else {
                messages_to_process.push(msg);
            }
        }
    }

    // Process tool execution counter first
    if any_tool_executed {
        app.add_tool_use();
        app.last_message_time = std::time::Instant::now();
    }

    // Process all collected messages
    let has_messages = !messages_to_process.is_empty();
    let mut any_completion = false;

    for msg in &messages_to_process {
        // Check for tool completion message
        let is_tool_completion =
            msg.contains("Result:") || (msg.contains("tool") && msg.contains("completed"));

        if is_tool_completion {
            any_completion = true;
        }

        // Add debug message if debug is enabled
        if app.debug_messages {
            app.messages
                .push(format!("DEBUG: Received agent message: {}", msg));
        }

        // Handle ANSI escape sequences by stripping them for storage but preserving their meaning
        let processed_msg = if msg.contains("\x1b[") {
            // Store a version without the ANSI codes for message matching but preserve the styling
            let clean_msg = msg.replace("\x1b[32m", "").replace("\x1b[0m", "");
            if app.debug_messages {
                app.messages.push(format!("[ansi_styled] {}", clean_msg));
            }
            clean_msg
        } else {
            msg.clone()
        };

        // Process the message
        process_message(app, &processed_msg)?;
    }

    // Update animation timestamp for tool completions
    if any_completion {
        app.last_message_time = std::time::Instant::now();
    }

    // Add auto-scroll marker if we processed any messages
    if has_messages || any_tool_executed {
        // Add one auto-scroll marker for each message (to ensure proper scroll amount)
        for _ in 0..messages_to_process.len().max(1) {
            app.messages.push("_AUTO_SCROLL_".into());
        }
        // Don't force redraw - let the process_auto_scroll function handle it
    }

    Ok(())
}

/// Process auto-scroll markers in messages without forcing redraws
fn process_auto_scroll(
    app: &mut App,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    // Check if we need to auto-scroll after processing messages
    let auto_scroll_count = app
        .messages
        .iter()
        .filter(|m| *m == "_AUTO_SCROLL_")
        .count();

    // Only process if there are markers
    if auto_scroll_count > 0 {
        // Remove auto-scroll markers
        app.messages.retain(|m| m != "_AUTO_SCROLL_");

        // Force content to be at the bottom - this ensures we always see new content
        app.message_scroll.follow_bottom = true;
        app.auto_scroll_to_bottom();

        // Force multiple immediate redraws with short delay to ensure content is visible
        // This helps overcome timing issues with terminal rendering
        for _ in 0..2 {
            terminal.draw(|f| ui(f, app))?;
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    Ok(())
}

/// Handle Left arrow key for cursor movement
fn handle_left_key(
    mut app: &mut App,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    match app.state {
        AppState::Chat | AppState::ApiKeyInput => {
            // Use textarea's built-in input method with proper conversion
            app.textarea.input(Input {
                key: Key::Left,
                ctrl: false,
                alt: false,
                shift: false,
            });

            // Update legacy cursor position for compatibility
            app.input = app.textarea.lines().join("\n");
            if app.cursor_position > 0 {
                app.cursor_position -= 1;
            }

            terminal.draw(|f| ui(f, &mut app))?;
        }
        _ => {}
    }
    Ok(())
}

/// Handle Right arrow key for cursor movement
fn handle_right_key(
    mut app: &mut App,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    match app.state {
        AppState::Chat | AppState::ApiKeyInput => {
            // Move cursor right using proper tui-textarea input
            app.textarea.input(Input {
                key: Key::Right,
                ctrl: false,
                alt: false,
                shift: false,
            });

            // Update legacy cursor position for compatibility
            app.input = app.textarea.lines().join("\n");
            if app.cursor_position < app.input.len() {
                app.cursor_position += 1;
            }

            terminal.draw(|f| ui(f, &mut app))?;
        }
        _ => {}
    }
    Ok(())
}

/// Process keyboard events
fn process_key_event(
    mut app: &mut App,
    key: KeyCode,
    modifiers: KeyModifiers,
    tx: &mpsc::Sender<String>,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    // Handle paste shortcuts - relies on terminal emulator's built-in paste support
    // Most terminals automatically handle paste operations by sending the text as if typed
    // We don't need to explicitly implement clipboard access, as the terminal will send
    // each character of the pasted content through the normal input channel
    // Handle permission response first if permission is required
    if app.permission_required {
        match key {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                // Grant permission
                app.handle_permission_response(true);
                app.permission_required = false;
                terminal.draw(|f| ui(f, &mut app))?;
                return Ok(());
            }
            KeyCode::Char('n') | KeyCode::Char('N') => {
                // Deny permission
                app.handle_permission_response(false);
                app.permission_required = false;
                terminal.draw(|f| ui(f, &mut app))?;
                return Ok(());
            }
            KeyCode::Esc => {
                // Cancel permission dialog (treat as deny)
                app.handle_permission_response(false);
                app.permission_required = false;
                terminal.draw(|f| ui(f, &mut app))?;
                return Ok(());
            }
            _ => return Ok(()), // Ignore other keys while permission dialog is active
        }
    }

    // Normal key handling if no permission dialog
    match key {
        KeyCode::Esc => {
            if app.debug_messages {
                app.messages.push("DEBUG: Esc pressed, exiting".into());
            }
            app.state = AppState::Error("quit".into());
        }
        KeyCode::Enter => {
            // Enhanced handling of newlines and Enter key
            if app.state == AppState::Chat {
                if modifiers.contains(KeyModifiers::SHIFT) || modifiers.contains(KeyModifiers::ALT)
                {
                    // Shift+Enter or Alt+Enter directly inserts a newline
                    // Using input method to ensure proper handling by tui-textarea
                    app.textarea.input(Input {
                        key: Key::Enter,
                        ctrl: false,
                        alt: modifiers.contains(KeyModifiers::ALT),
                        shift: modifiers.contains(KeyModifiers::SHIFT),
                    });

                    // Update the legacy input for compatibility
                    app.input = app.textarea.lines().join("\n");

                    // Force immediate redraw to update input box size and cursor position
                    terminal.draw(|f| ui(f, &mut app))?;
                } else {
                    // Regular Enter handling
                    handle_enter_key(app, tx, terminal)?;
                }
            } else {
                // Regular Enter handling for non-Chat states
                handle_enter_key(app, tx, terminal)?;
            }
        }
        KeyCode::Down => {
            if modifiers.contains(KeyModifiers::SHIFT) {
                // Shift+Down scrolls task list down
                handle_task_scroll_down(app, terminal)?
            } else {
                handle_down_key(app, terminal)?
            }
        }
        KeyCode::Tab => handle_tab_key(app, terminal)?,
        KeyCode::Up => {
            if modifiers.contains(KeyModifiers::SHIFT) {
                // Shift+Up scrolls task list up
                handle_task_scroll_up(app, terminal)?
            } else {
                handle_up_key(app, terminal)?
            }
        }
        KeyCode::Left => handle_left_key(app, terminal)?,
        KeyCode::Right => handle_right_key(app, terminal)?,
        KeyCode::BackTab => handle_backtab_key(app, terminal)?,
        KeyCode::Char(c) => handle_char_key(app, c, modifiers, terminal)?,
        KeyCode::Backspace => handle_backspace_key(app, terminal)?,
        KeyCode::PageUp => handle_page_up_key(app, terminal)?,
        KeyCode::PageDown => handle_page_down_key(app, terminal)?,
        KeyCode::Home => handle_home_key(app, terminal)?,
        KeyCode::End => handle_end_key(app, terminal)?,
        _ => {}
    }

    Ok(())
}

/// Handle Enter key in different application states
fn handle_enter_key(
    mut app: &mut App,
    tx: &mpsc::Sender<String>,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    if app.debug_messages {
        app.messages.push("DEBUG: Enter key pressed".into());
    }

    match app.state {
        AppState::Setup => {
            app.messages.push("DEBUG: Starting model setup...".into());
            terminal.draw(|f| ui(f, &mut app))?;

            if let Err(e) = app.setup_models(tx.clone()) {
                app.messages.push(format!("ERROR: Setup failed: {}", e));
            }
            terminal.draw(|f| ui(f, &mut app))?;
        }
        AppState::ApiKeyInput => {
            // Get input from the textarea
            let api_key = app.textarea.lines().join("\n");
            // Clear the textarea
            app.textarea.delete_line_by_end();

            if !api_key.is_empty() {
                app.messages
                    .push("DEBUG: API key entered, continuing setup...".into());

                // Set the API key and return to setup state
                app.api_key = Some(api_key);
                app.state = AppState::Setup;

                // When returning to regular input, unmask characters (use space as "no mask")
                app.textarea.set_mask_char(' ');

                // Continue with model setup using the provided API key
                if let Err(e) = app.setup_models(tx.clone()) {
                    app.messages.push(format!("ERROR: Setup failed: {}", e));
                }
                terminal.draw(|f| ui(f, &mut app))?;
            } else {
                app.messages
                    .push("API key cannot be empty. Please enter your Anthropic API key...".into());
            }
        }
        AppState::Chat => {
            // First check if we're in command mode
            if app.command_mode {
                // Try to execute the command
                let cmd_executed = app.execute_command();

                // Clear the textarea after executing the command
                app.textarea.delete_line_by_end();
                app.textarea.delete_line_by_head();
                app.input.clear(); // Clear legacy input for compatibility
                app.command_mode = false;
                app.show_command_menu = false;

                // Skip model querying if we executed a command
                if cmd_executed {
                    // Need to redraw to clear command menu
                    terminal.draw(|f| ui(f, &mut app))?;
                    return Ok(());
                }
            }

            // Get the input from the textarea
            let input = app.textarea.lines().join("\n");

            // Clear the textarea after submitting
            while !app.textarea.is_empty() {
                app.textarea.delete_line_by_end();
                app.textarea.delete_line_by_head();
                if !app.textarea.is_empty() {
                    // Move to next line if there are more lines
                    app.textarea.input(Input {
                        key: Key::Down,
                        ctrl: false,
                        alt: false,
                        shift: false,
                    });
                }
            }

            // Update legacy input field for compatibility
            app.input.clear();

            if !input.is_empty() {
                // No debug output needed here

                // Create a new task for this query
                let _task_id = app.create_task(&input);

                // Estimate input tokens - a basic approximation is 4 characters per token
                let estimated_input_tokens = (input.len() / 4) as u32;
                app.add_input_tokens(estimated_input_tokens);

                // Add user message with preserved newlines
                app.messages.push(format!("> {}", input));

                // No thinking message needed - async tasks will show their own progress
                // Force immediate redraw to show the input has been received
                app.auto_scroll_to_bottom();
                terminal.draw(|f| ui(f, &mut app))?;

                // Update the last query time
                app.last_query_time = std::time::Instant::now();

                // CRITICAL FIX: We need to process tool messages BEFORE showing the final answer
                // The key issue is that we need to continue processing agent messages while
                // the query is being executed, but before we get the final result.

                // Force UI refresh for better UX
                app.auto_scroll_to_bottom();
                terminal.draw(|f| ui(f, &mut app))?;

                // Start the model query - this initiates tool execution, but doesn't
                // return until all tool execution is complete
                app.tool_execution_in_progress = true; // Set this manually to ensure proper animation

                // Process a batch of agent messages before starting the query
                // to make sure the UI is set up properly
                process_agent_messages(app, terminal)?;
                terminal.draw(|f| ui(f, &mut app))?;

                // Process agent messages in a special loop to ensure they're displayed
                // BEFORE we get the final result
                let start_time = std::time::Instant::now();
                let timeout = Duration::from_secs(2); // Short timeout to ensure tools start processing

                // First phase - wait for the first tool message to appear
                // This ensures we see "tool executing" before we see results
                while start_time.elapsed() < timeout {
                    // Check for and process agent messages
                    process_agent_messages(app, terminal)?;
                    process_auto_scroll(app, terminal)?;

                    // Redraw the UI to show any updates
                    terminal.draw(|f| ui(f, &mut app))?;

                    // If we've processed any tool messages, we can start the query
                    if app.tool_execution_in_progress {
                        // Give tools a chance to execute and display
                        std::thread::sleep(Duration::from_millis(200));
                        break;
                    }

                    // Brief pause to avoid busy-waiting
                    std::thread::sleep(Duration::from_millis(50));
                }

                // Now execute the actual query and get the final result
                // This ensures all tool messages are displayed BEFORE we get the final result
                let result = match app.parse_code_mode {
                    // If we're in parse_code mode, this input is a file path to parse
                    true => {
                        app.parse_code_mode = false; // Turn off the mode after processing
                        app.handle_parse_code_command(&input)
                    }
                    // Otherwise, normal query
                    false => app.query_model(&input),
                };

                // Final phase - make sure we've displayed all tool messages
                let final_timeout = Duration::from_millis(500);
                let final_start = std::time::Instant::now();

                while final_start.elapsed() < final_timeout {
                    // Process any remaining agent messages
                    process_agent_messages(app, terminal)?;
                    process_auto_scroll(app, terminal)?;

                    // Redraw to ensure tools are displayed
                    terminal.draw(|f| ui(f, &mut app))?;

                    // Brief pause
                    std::thread::sleep(Duration::from_millis(50));
                }

                // Process the final result
                match result {
                    Ok(response_string) => {
                        // Remove any thinking messages
                        if let Some(last) = app.messages.last() {
                            if last == "Thinking..."
                                || last.starts_with("[thinking]")
                                || last.starts_with("⚪ Processing")
                            {
                                app.messages.pop();
                            }
                        }

                        // Process and format the response for better display
                        format_and_display_response(app, &response_string);

                        // Complete the task with estimated output tokens
                        let estimated_output_tokens = (response_string.len() / 4) as u32;
                        app.complete_current_task(estimated_output_tokens);

                        // Force scrolling to the bottom to show the new response
                        app.auto_scroll_to_bottom();
                    }
                    Err(e) => {
                        // Remove any thinking messages
                        if let Some(last) = app.messages.last() {
                            if last == "Thinking..."
                                || last.starts_with("[thinking]")
                                || last.starts_with("⚪ Processing")
                            {
                                app.messages.pop();
                            }
                        }

                        // Mark the task as failed
                        app.fail_current_task(&e.to_string());

                        app.messages.push(format!("Error: {}", e));
                        app.auto_scroll_to_bottom();
                    }
                }

                // Final redraw to ensure everything is displayed
                terminal.draw(|f| ui(f, &mut app))?;

                // Make sure to redraw after getting a response
                terminal.draw(|f| ui(f, &mut app))?;
            }
        }
        AppState::Error(_) => {
            app.state = AppState::Setup;
            app.error_message = None;
        }
    }
    terminal.draw(|f| ui(f, &mut app))?;

    Ok(())
}

/// Format and display a model response
fn format_and_display_response(app: &mut App, response: &str) {
    // Split long responses into multiple messages if needed
    let max_line_length = 80; // Reasonable line length for TUI display

    if response.contains('\n') {
        // For multi-line responses (code or structured content)
        // Add an empty line before the response for readability
        app.messages.push("".to_string());

        // Split by line to preserve formatting
        for line in response.lines() {
            // For very long lines, add wrapping
            if line.len() > max_line_length {
                // Simple wrapping at character boundaries
                // Use integer division that rounds up (equivalent to ceiling division)
                // Skip clippy suggestion as div_ceil might not be available in all Rust versions
                #[allow(clippy::manual_div_ceil)]
                let chunk_count = (line.len() + max_line_length - 1) / max_line_length;
                for i in 0..chunk_count {
                    let start = i * max_line_length;
                    let end = std::cmp::min(start + max_line_length, line.len());
                    if start < line.len() {
                        app.messages.push(line[start..end].to_string());
                    }
                }
            } else {
                app.messages.push(line.to_string());
            }
        }

        // Add another empty line after for readability
        app.messages.push("".to_string());
    } else {
        // For single-line responses, add directly
        app.messages.push(response.to_string());
    }
}

/// Handle Down key in different application states
fn handle_down_key(
    mut app: &mut App,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    match app.state {
        AppState::Setup => {
            app.select_next_model();
            app.messages.push("DEBUG: Selected next model".into());
            terminal.draw(|f| ui(f, &mut app))?;
        }
        AppState::Chat => {
            // Navigate commands in command mode
            if app.show_command_menu {
                app.select_next_command();
                terminal.draw(|f| ui(f, &mut app))?;
            }
            // When not in command mode, handle multiline navigation with TextArea
            else if !app.textarea.is_empty() {
                // Move down using tui-textarea method
                app.textarea.input(Input {
                    key: Key::Down,
                    ctrl: false,
                    alt: false,
                    shift: false,
                });

                // Update legacy input and cursor for compatibility
                app.input = app.textarea.lines().join("\n");

                // Force redraw to update cursor position
                terminal.draw(|f| ui(f, &mut app))?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Handle Tab key in different application states
fn handle_tab_key(
    mut app: &mut App,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    match app.state {
        AppState::Setup => {
            app.select_next_model();
            app.messages.push("DEBUG: Selected next model".into());
            terminal.draw(|f| ui(f, &mut app))?;
        }
        AppState::Chat => {
            // Auto-complete command if in command mode
            if app.show_command_menu {
                let filtered = app.filtered_commands();
                if !filtered.is_empty() && app.selected_command < filtered.len() {
                    // Auto-complete with selected command
                    app.input = filtered[app.selected_command].name.clone();
                    app.show_command_menu = true;
                    app.command_mode = true;
                }
                terminal.draw(|f| ui(f, &mut app))?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Handle Up key in different application states
fn handle_up_key(
    mut app: &mut App,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    match app.state {
        AppState::Setup => {
            app.select_prev_model();
            app.messages.push("DEBUG: Selected previous model".into());
            terminal.draw(|f| ui(f, &mut app))?;
        }
        AppState::Chat => {
            // Navigate commands in command mode
            if app.show_command_menu {
                app.select_prev_command();
                terminal.draw(|f| ui(f, &mut app))?;
            }
            // When not in command mode, handle multiline navigation with TextArea
            else if !app.textarea.is_empty() {
                // Move up using tui-textarea method
                app.textarea.input(Input {
                    key: Key::Up,
                    ctrl: false,
                    alt: false,
                    shift: false,
                });

                // Update legacy input and cursor for compatibility
                app.input = app.textarea.lines().join("\n");

                // Force redraw to update cursor position
                terminal.draw(|f| ui(f, &mut app))?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Handle BackTab key in different application states
fn handle_backtab_key(
    mut app: &mut App,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    if let AppState::Setup = app.state {
        app.select_prev_model();
        app.messages.push("DEBUG: Selected previous model".into());
        terminal.draw(|f| ui(f, &mut app))?;
    }
    Ok(())
}

/// Handle character key in different application states
fn handle_char_key(
    mut app: &mut App,
    c: char,
    modifiers: KeyModifiers,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    match app.state {
        AppState::Chat | AppState::ApiKeyInput => {
            // Special handling for '?' to toggle shortcut display
            if app.state == AppState::Chat && c == '?' && app.textarea.is_empty() {
                // Toggle detailed shortcuts display and don't add the character
                app.show_detailed_shortcuts = !app.show_detailed_shortcuts;
                terminal.draw(|f| ui(f, &mut app))?;
                return Ok(());
            }

            // Special handling for 'j' key with Ctrl modifier to add a newline (common shortcut in many editors)
            if app.state == AppState::Chat
                && (c == 'j' || c == 'J')
                && modifiers.contains(KeyModifiers::CONTROL)
            {
                // Insert a newline character
                app.textarea.insert_char('\n');

                // Update legacy input for compatibility
                app.input = app.textarea.lines().join("\n");

                // Update cursor position for compatibility
                let (x, y) = app.textarea.cursor();
                app.cursor_position = app
                    .textarea
                    .lines()
                    .iter()
                    .take(y)
                    .map(|line| line.len() + 1) // +1 for newline
                    .sum::<usize>()
                    + x;

                // Force immediate redraw to update input box size and cursor position
                terminal.draw(|f| ui(f, &mut app))?;
                return Ok(());
            }

            // Insert the character using proper input method
            app.textarea.input(Input {
                key: Key::Char(c),
                ctrl: modifiers.contains(KeyModifiers::CONTROL),
                alt: modifiers.contains(KeyModifiers::ALT),
                shift: modifiers.contains(KeyModifiers::SHIFT),
            });

            // Update legacy input field for compatibility
            app.input = app.textarea.lines().join("\n");

            // Update cursor position for compatibility
            let (x, y) = app.textarea.cursor();
            // Calculate cursor position based on line length up to the current line plus current position
            app.cursor_position = app
                .textarea
                .lines()
                .iter()
                .take(y)
                .map(|line| line.len() + 1) // +1 for newline
                .sum::<usize>()
                + x;

            // Check if we're entering command mode with the / character
            if app.state == AppState::Chat
                && c == '/'
                && app.textarea.lines().len() == 1
                && app.textarea.lines()[0] == "/"
            {
                app.command_mode = true;
                app.show_command_menu = true;
                app.selected_command = 0;
                // Hide detailed shortcuts when typing /
                app.show_detailed_shortcuts = false;
            } else if app.command_mode {
                // Update command mode state
                app.check_command_mode();
            } else {
                // Hide detailed shortcuts when typing anything else
                app.show_detailed_shortcuts = false;
            }

            terminal.draw(|f| ui(f, &mut app))?;
        }
        _ => {}
    }
    Ok(())
}

/// Handle backspace key in different application states
fn handle_backspace_key(
    mut app: &mut App,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    match app.state {
        AppState::Chat | AppState::ApiKeyInput => {
            // Delete the character before the cursor using proper input method
            app.textarea.input(Input {
                key: Key::Backspace,
                ctrl: false,
                alt: false,
                shift: false,
            });

            // Update legacy input field for compatibility
            app.input = app.textarea.lines().join("\n");

            // Update legacy cursor position for compatibility
            if app.cursor_position > 0 {
                app.cursor_position -= 1;
            }

            // Check if we've exited command mode
            if app.state == AppState::Chat {
                app.check_command_mode();
            }

            terminal.draw(|f| ui(f, &mut app))?;
        }
        _ => {}
    }
    Ok(())
}

/// Handle PageUp key for scrolling
fn handle_page_up_key(
    mut app: &mut App,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    if let AppState::Chat = app.state {
        // Turn off auto-follow when manually scrolling up
        app.message_scroll.follow_bottom = false;

        // Use page_up method for better scrolling behavior based on viewport size
        app.message_scroll.page_up();

        // Update legacy scroll position to match ScrollState
        app.scroll_position = app.message_scroll.position;

        // Immediately redraw to show new scroll position
        terminal.draw(|f| ui(f, &mut app))?;
    }
    Ok(())
}

/// Handle PageDown key for scrolling
fn handle_page_down_key(
    mut app: &mut App,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    if let AppState::Chat = app.state {
        // Use page_down method for better scrolling behavior based on viewport size
        app.message_scroll.page_down();

        // Update legacy scroll position to match ScrollState
        app.scroll_position = app.message_scroll.position;

        // Enable auto-follow if we've reached the bottom
        if app.message_scroll.position >= app.message_scroll.max_scroll() {
            app.message_scroll.follow_bottom = true;
        }

        // Immediately redraw to show new scroll position
        terminal.draw(|f| ui(f, &mut app))?;
    }
    Ok(())
}

/// Handle task list scrolling with Shift+Up
fn handle_task_scroll_up(
    mut app: &mut App,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    if let AppState::Chat = app.state {
        // Use the task scroll state
        app.task_scroll.scroll_up(1);

        // Update legacy scroll position to match ScrollState
        app.task_scroll_position = app.task_scroll.position;

        terminal.draw(|f| ui(f, &mut app))?;
    }
    Ok(())
}

/// Handle task list scrolling with Shift+Down
fn handle_task_scroll_down(
    mut app: &mut App,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    if let AppState::Chat = app.state {
        // Use the task scroll state
        app.task_scroll.scroll_down(1);

        // Update legacy scroll position to match ScrollState
        app.task_scroll_position = app.task_scroll.position;

        terminal.draw(|f| ui(f, &mut app))?;
    }
    Ok(())
}

/// Handle Home key for scrolling to top and moving cursor to start of input
fn handle_home_key(
    mut app: &mut App,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    match app.state {
        AppState::Chat => {
            if app.textarea.is_empty() {
                // If no input, use Home to scroll the message window to top
                // Turn off auto-follow when manually scrolling to top
                app.message_scroll.follow_bottom = false;
                app.message_scroll.scroll_to_top();

                // Update legacy scroll position to match ScrollState
                app.scroll_position = app.message_scroll.position;

                // Immediately redraw to show scroll position change
                terminal.draw(|f| ui(f, &mut app))?;
            } else {
                // Move to start of line
                app.textarea.input(Input {
                    key: Key::Home,
                    ctrl: false,
                    alt: false,
                    shift: false,
                });

                // Update legacy cursor position for compatibility
                app.input = app.textarea.lines().join("\n");
                let (x, _y) = app.textarea.cursor();
                app.cursor_position = x;

                // Redraw to update cursor position
                terminal.draw(|f| ui(f, &mut app))?;
            }
        }
        AppState::ApiKeyInput => {
            // Move cursor to start of input
            app.textarea.input(Input {
                key: Key::Home,
                ctrl: false,
                alt: false,
                shift: false,
            });
            app.cursor_position = 0; // Update legacy cursor position
            terminal.draw(|f| ui(f, &mut app))?;
        }
        _ => {}
    }
    Ok(())
}

/// Handle End key for scrolling to bottom and moving cursor to end of input
fn handle_end_key(
    mut app: &mut App,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    match app.state {
        AppState::Chat => {
            if app.textarea.is_empty() {
                // If no input, use End to scroll the message window to bottom
                // Enable auto-follow when scrolling to bottom
                app.message_scroll.follow_bottom = true;
                app.message_scroll.scroll_to_bottom();

                // Update legacy scroll position to match ScrollState
                app.scroll_position = app.message_scroll.position;

                // Immediately redraw to show scroll position change
                terminal.draw(|f| ui(f, &mut app))?;
            } else {
                // Move to end of line
                app.textarea.input(Input {
                    key: Key::End,
                    ctrl: false,
                    alt: false,
                    shift: false,
                });

                // Update legacy cursor position for compatibility
                app.input = app.textarea.lines().join("\n");
                app.cursor_position = app.input.len();

                // Redraw to update cursor position
                terminal.draw(|f| ui(f, &mut app))?;
            }
        }
        AppState::ApiKeyInput => {
            // Move cursor to end of input
            app.textarea.input(Input {
                key: Key::End,
                ctrl: false,
                alt: false,
                shift: false,
            });

            // Update legacy cursor position
            app.input = app.textarea.lines().join("\n");
            app.cursor_position = app.input.len();

            terminal.draw(|f| ui(f, &mut app))?;
        }
        _ => {}
    }
    Ok(())
}
