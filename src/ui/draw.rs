use crate::app::commands::CommandHandler;
use crate::app::models::ModelManager;
use crate::app::state::{App, AppState};
use crate::ui::components::*;
use crate::ui::styles::AppStyles;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Padding, Paragraph},
    Frame,
};

/// Add a prompt symbol in front of the textarea
fn add_prompt(f: &mut Frame, area: Rect, prompt_text: &str) {
    // Create a prompt character in blue
    let prompt = Span::styled(prompt_text, AppStyles::prompt_symbol());

    // Create a paragraph with just the prompt
    let prompt_para = Paragraph::new(Line::from(vec![prompt]));

    // Calculate the actual text area inside the block borders
    // Add 1 for the border and 1 for inner padding
    let prompt_area = Rect {
        x: area.x + 1,
        y: area.y + 1,
        width: prompt_text.len() as u16,
        height: 1,
    };

    // Render the prompt over the textarea
    f.render_widget(prompt_para, prompt_area);
}

/// Main UI rendering function, dispatches to specific screen renderers
pub fn ui(f: &mut Frame, app: &mut App) {
    // Calculate time elapsed since last message for animation effects
    // This is used to update app.last_message_time for continuous animation
    let _animation_active =
        app.last_message_time.elapsed() < std::time::Duration::from_millis(1000);

    // Update last message time for continuous animation when tool execution is in progress
    if app.tool_execution_in_progress || app.agent_progress_rx.is_some() {
        // Only update the timestamp if we're actively processing
        // This will keep the animation going for active tasks
        app.last_message_time = std::time::Instant::now();
    }

    match app.state {
        AppState::Setup => draw_setup(f, app),
        AppState::ApiKeyInput => draw_api_key_input(f, app),
        AppState::Chat => draw_chat(f, app),
        AppState::Error(ref error_msg) => draw_error(f, error_msg),
    }

    // Note: We no longer need to mutate app here as this is handled
    // in the events.rs file before calling ui.

    // Draw permission dialog over other UI elements when needed
    if app.permission_required && app.pending_tool.is_some() {
        draw_permission_dialog(f, app);
    }
}

/// Draw setup screen with model selection
pub fn draw_setup(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(3)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(3),
        ])
        .split(f.area());

    // Title with version
    let version = env!("CARGO_PKG_VERSION");
    let title = Paragraph::new(format!("oli v{} Setup Assistant", version))
        .style(AppStyles::title())
        .alignment(Alignment::Center);
    f.render_widget(title, chunks[0]);

    // Model list
    let models_list = create_model_list(app);
    f.render_widget(models_list, chunks[1]);

    // Progress
    let progress_bar = create_progress_display(app);
    f.render_widget(progress_bar, chunks[2]);
}

/// Draw API key input screen
pub fn draw_api_key_input(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(3)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(3),
        ])
        .split(f.area());

    // Determine title based on selected model
    let version = env!("CARGO_PKG_VERSION");
    let title_text = match app.current_model().name.as_str() {
        "GPT-4o" => format!("Oli v{} - OpenAI API Key Setup", version),
        name if name.contains("Local") => format!("Oli v{} - Local Model Setup", version),
        _ => format!("Oli v{} - Anthropic API Key Setup", version),
    };

    let title = Paragraph::new(title_text)
        .style(AppStyles::title())
        .alignment(Alignment::Center);
    f.render_widget(title, chunks[0]);

    // Information area showing API key requirements
    let info = create_api_key_info(app);
    f.render_widget(info, chunks[1]);

    // Create a masked input block for API keys
    let input_block = Block::default()
        .borders(Borders::ALL)
        .title(" API Key ")
        .title_alignment(Alignment::Left)
        .border_style(AppStyles::border());

    // Set the block for textarea and mask characters
    app.textarea.set_block(input_block);
    app.textarea.set_mask_char('*'); // Mask input with asterisks

    // Create custom block with padding for prompt
    let input_block_with_padding = Block::default()
        .borders(Borders::ALL)
        .title(" API Key ")
        .title_alignment(Alignment::Left)
        .border_style(AppStyles::border())
        .padding(Padding::new(3, 1, 0, 0)); // Extra left padding for prompt

    // Set the block with padding for textarea
    app.textarea.set_block(input_block_with_padding);

    // Render the masked textarea
    f.render_widget(&app.textarea, chunks[2]);

    // Add a blue prompt in front of the input
    add_prompt(f, chunks[2], ">");
}

/// Draw chat screen with message history and input
pub fn draw_chat(f: &mut Frame, app: &mut App) {
    // Split the screen into two horizontal panes: left (task list) and right (chat)
    let horizontal_split = Layout::default()
        .direction(Direction::Horizontal)
        .margin(1)
        .constraints([
            Constraint::Percentage(25), // Task list - 25% of the screen
            Constraint::Percentage(75), // Chat area - 75% of the screen
        ])
        .split(f.area());

    // Left pane: Task list
    let tasks_area = horizontal_split[0];
    let task_list = create_task_list(app, tasks_area);
    f.render_widget(task_list, tasks_area);

    // Right pane: Chat area
    let chat_area = horizontal_split[1];

    // Count lines in textarea to determine input box height
    let line_count = app.textarea.lines().len();

    // Calculate how many wrapped lines we might need based on the terminal width
    // This helps with large pastes that would otherwise overflow horizontally
    let terminal_width = chat_area.width.saturating_sub(4); // Account for borders and padding
    let wrapped_line_count = app
        .textarea
        .lines()
        .iter()
        .map(|line| {
            if line.is_empty() {
                1 // Empty lines still need one line
            } else {
                // Calculate how many lines this content would wrap to
                // Using div_ceil pattern for proper ceiling division
                (line.len() as u16)
                    .saturating_add(terminal_width)
                    .saturating_sub(1)
                    / terminal_width
            }
        })
        .sum::<u16>() as usize;

    // Use the larger of actual lines or wrapped lines to determine height
    let _effective_line_count = line_count.max(wrapped_line_count);

    // Calculate base input height (min 3, grows with lines but caps at half the available height)
    // First, estimate the total available height (area height minus margins and other UI elements)
    let estimated_available_height = chat_area.height.saturating_sub(4); // Subtract margins and status bar

    // Fixed input height at 20% of the chat area
    let max_input_height = estimated_available_height / 5; // 20% of available height
    let base_input_height = 3.max(max_input_height as usize);

    let input_height = if app.show_command_menu {
        // Increase the input area height to make room for the command menu
        let cmd_count = app.filtered_commands().len();
        // Add command menu height (up to 5) to the base input height
        base_input_height + cmd_count.min(5)
    } else {
        base_input_height // Use calculated height based on content
    };

    // Calculate height for shortcuts area - only show when textarea is empty
    let shortcuts_height = if app.textarea.is_empty() {
        if app.show_detailed_shortcuts {
            4 // Height for detailed shortcuts panel (increased for new shortcut)
        } else if app.show_shortcuts_hint {
            1 // Height for shortcut hint
        } else {
            0 // No height when shortcuts are disabled
        }
    } else {
        0 // No height when anything is typed in the input
    };

    // Split the chat area vertically into status bar, message history, and input
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),                   // Status bar
            Constraint::Min(5),                      // Chat history (expandable)
            Constraint::Length(input_height as u16), // Input area (with variable height for command menu)
            Constraint::Length(shortcuts_height),    // Shortcuts area (variable height)
        ])
        .split(chat_area);

    // Status bar showing model info and scroll position
    let status_bar = create_status_bar(app);
    let status_bar_widget = Paragraph::new(status_bar).style(Style::default());
    f.render_widget(status_bar_widget, chunks[0]);

    // Messages history or logs based on app.show_logs flag
    if app.show_logs {
        // Render logs instead of messages
        let logs_widget = create_log_list(app, chunks[1]);
        f.render_widget(logs_widget, chunks[1]);
    } else {
        // Render normal messages
        let messages_widget = create_message_list(app, chunks[1]);
        f.render_widget(messages_widget, chunks[1]);
    }

    // Split the input area if command menu is visible
    if app.show_command_menu {
        // Split the input area into the input box and command menu
        let input_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),                                           // Input box
                Constraint::Length(app.filtered_commands().len().min(5) as u16), // Command menu (max 5 items)
            ])
            .split(chunks[2]);

        // Create input block with title and padding for prompt
        let input_block = Block::default()
            .borders(Borders::ALL)
            .title(" Input (Type / for commands) ")
            .title_alignment(Alignment::Left)
            .border_style(AppStyles::border())
            .padding(Padding::new(3, 1, 0, 0)); // Extra left padding for prompt

        // Set the block for the textarea
        app.textarea.set_block(input_block);

        // Render the textarea
        f.render_widget(&app.textarea, input_chunks[0]);

        // Add a blue prompt in front of the input
        add_prompt(f, input_chunks[0], ">");

        // Commands menu as a list
        let commands_list = create_command_menu(app);
        f.render_widget(commands_list, input_chunks[1]);
    } else {
        // Create input block with title and padding for prompt
        let input_block = Block::default()
            .borders(Borders::ALL)
            .title(" Input (Type / for commands) ")
            .title_alignment(Alignment::Left)
            .border_style(AppStyles::border())
            .padding(Padding::new(2, 1, 0, 0)); // Extra left padding for prompt

        // Set the block for the textarea
        app.textarea.set_block(input_block);

        // Render the textarea with its block
        f.render_widget(&app.textarea, chunks[2]);

        // Add a blue prompt in front of the input
        add_prompt(f, chunks[2], ">");
    }

    // Render shortcuts panel if needed
    if shortcuts_height > 0 {
        let shortcuts_panel = create_shortcuts_panel(app);
        f.render_widget(shortcuts_panel, chunks[3]);
    }
}

/// Draw error screen
pub fn draw_error(f: &mut Frame, error_msg: &str) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(3)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(3),
        ])
        .split(f.area());

    let title = Paragraph::new("Error Occurred")
        .style(Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center);
    f.render_widget(title, chunks[0]);

    let error_text = Paragraph::new(error_msg)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Error Details ")
                .title_alignment(Alignment::Left)
                .border_style(Style::default().fg(Color::Red))
                .padding(Padding::new(1, 1, 0, 0)),
        )
        .style(Style::default().fg(Color::Red))
        .wrap(ratatui::widgets::Wrap { trim: true });
    f.render_widget(error_text, chunks[1]);

    let instruction = Paragraph::new("Press Enter to return to setup or Esc to exit")
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Yellow))
                .padding(Padding::new(0, 0, 0, 0)),
        )
        .style(Style::default().fg(Color::Yellow))
        .alignment(Alignment::Center);
    f.render_widget(instruction, chunks[2]);
}

// Cursor positioning now handled by tui-textarea component

/// Draw permission dialog over the current UI
pub fn draw_permission_dialog(f: &mut Frame, app: &App) {
    // Calculate dialog size and position (centered)
    let area = f.area();
    let width = std::cmp::min(90, area.width.saturating_sub(8));

    // Adjust height based on whether we have a diff preview
    let has_diff = app
        .pending_tool
        .as_ref()
        .and_then(|pt| pt.diff_preview.as_ref())
        .is_some();

    // Minimum height with or without diff preview
    let height = if has_diff {
        std::cmp::min(30, area.height.saturating_sub(4))
    } else {
        10
    };

    let x = (area.width - width) / 2;
    let y = (area.height - height) / 2;

    let dialog_area = Rect::new(x, y, width, height);

    // Create the dialog content
    let dialog = create_permission_dialog(app, dialog_area);

    f.render_widget(dialog, dialog_area);

    // Create content inside the dialog
    let inner_area = Rect {
        x: dialog_area.x + 1,
        y: dialog_area.y + 1,
        width: dialog_area.width.saturating_sub(2),
        height: dialog_area.height.saturating_sub(2),
    };

    // Display tool info
    let info = create_permission_content(app);

    f.render_widget(info, inner_area);
}
