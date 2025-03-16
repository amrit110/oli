use crate::app::{App, AppState};
use anyhow::Result;
use crossterm::{
    cursor::{Hide, Show},
    event::{Event, KeyCode},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame, Terminal,
};
use std::{io, time::Duration};
use unicode_width::UnicodeWidthStr;

struct TerminalGuard;

impl TerminalGuard {
    fn new() -> Result<Self> {
        enable_raw_mode()?;
        crossterm::execute!(io::stdout(), EnterAlternateScreen, Hide)?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = crossterm::execute!(io::stdout(), LeaveAlternateScreen, Show);
        let _ = disable_raw_mode();
    }
}

pub fn run_app() -> Result<()> {
    let _guard = TerminalGuard::new()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    let mut app = App::new();
    initialize_messages(&mut app);
    app.messages
        .push("DEBUG: Application started. Press Enter to begin setup.".into());

    let (tx, rx) = std::sync::mpsc::channel::<String>();

    terminal.draw(|f| ui(f, &app))?;

    while app.state != AppState::Error("quit".into()) {
        // Always redraw if download is active, to show progress
        if app.download_active {
            terminal.draw(|f| ui(f, &app))?;
        }

        // Check for messages from download thread
        while let Ok(msg) = rx.try_recv() {
            app.messages
                .push(format!("DEBUG: Received message: {}", msg));
            process_message(&mut app, &msg)?;
            terminal.draw(|f| ui(f, &app))?;
        }

        // Check for messages from agent progress
        let agent_messages = if let Some(ref agent_rx) = app.agent_progress_rx {
            let mut messages = Vec::new();
            while let Ok(msg) = agent_rx.try_recv() {
                app.messages
                    .push(format!("DEBUG: Received agent message: {}", msg));
                messages.push(msg);
            }
            messages
        } else {
            Vec::new()
        };

        // Process agent messages
        for msg in agent_messages {
            process_message(&mut app, &msg)?;
            terminal.draw(|f| ui(f, &app))?;
        }

        if crossterm::event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = crossterm::event::read()? {
                match key.code {
                    KeyCode::Esc => {
                        app.messages.push("DEBUG: Esc pressed, exiting".into());
                        break;
                    }
                    KeyCode::Enter => {
                        app.messages.push("DEBUG: Enter key pressed".into());

                        match app.state {
                            AppState::Setup => {
                                app.messages.push("DEBUG: Starting model setup...".into());
                                terminal.draw(|f| ui(f, &app))?;

                                if let Err(e) = app.setup_models(tx.clone()) {
                                    app.messages.push(format!("ERROR: Setup failed: {}", e));
                                }
                                terminal.draw(|f| ui(f, &app))?;
                            }
                            AppState::ApiKeyInput => {
                                let api_key = std::mem::take(&mut app.input);
                                if !api_key.is_empty() {
                                    app.messages
                                        .push("DEBUG: API key entered, continuing setup...".into());

                                    // Set the API key and return to setup state
                                    app.api_key = Some(api_key);
                                    app.state = AppState::Setup;

                                    // Continue with model setup using the provided API key
                                    if let Err(e) = app.setup_models(tx.clone()) {
                                        app.messages.push(format!("ERROR: Setup failed: {}", e));
                                    }
                                    terminal.draw(|f| ui(f, &app))?;
                                } else {
                                    app.messages.push("API key cannot be empty. Please enter your Anthropic API key...".into());
                                }
                            }
                            AppState::Chat => {
                                let input = std::mem::take(&mut app.input);
                                if !input.is_empty() {
                                    app.messages.push(format!("> {}", input));

                                    // Show a "thinking" message
                                    app.messages.push("Thinking...".into());
                                    terminal.draw(|f| ui(f, &app))?;

                                    // Update the last query time
                                    app.last_query_time = std::time::Instant::now();

                                    // Query the model
                                    match app.query_model(&input) {
                                        Ok(response) => {
                                            // Remove the thinking message
                                            if let Some(last) = app.messages.last() {
                                                if last == "Thinking..." {
                                                    app.messages.pop();
                                                }
                                            }

                                            // Format the response nicely
                                            let formatted_response = response;
                                            app.messages.push(formatted_response);

                                            // Auto-scroll to the bottom to show the new response
                                            app.auto_scroll_to_bottom();
                                        }
                                        Err(e) => {
                                            // Remove the thinking message
                                            if let Some(last) = app.messages.last() {
                                                if last == "Thinking..." {
                                                    app.messages.pop();
                                                }
                                            }
                                            app.messages.push(format!("Error: {}", e));
                                            app.auto_scroll_to_bottom();
                                        }
                                    }

                                    // Make sure to redraw after getting a response
                                    terminal.draw(|f| ui(f, &app))?;
                                }
                            }
                            AppState::Error(_) => {
                                app.state = AppState::Setup;
                                app.error_message = None;
                            }
                        }
                        terminal.draw(|f| ui(f, &app))?;
                    }
                    KeyCode::Down | KeyCode::Tab => {
                        if let AppState::Setup = app.state {
                            app.select_next_model();
                            app.messages.push("DEBUG: Selected next model".into());
                            terminal.draw(|f| ui(f, &app))?;
                        }
                    }
                    KeyCode::Up | KeyCode::BackTab => {
                        if let AppState::Setup = app.state {
                            app.select_prev_model();
                            app.messages.push("DEBUG: Selected previous model".into());
                            terminal.draw(|f| ui(f, &app))?;
                        }
                    }
                    KeyCode::Char(c) => match app.state {
                        AppState::Chat | AppState::ApiKeyInput => {
                            app.input.push(c);
                            terminal.draw(|f| ui(f, &app))?;
                        }
                        _ => {}
                    },
                    KeyCode::Backspace => match app.state {
                        AppState::Chat | AppState::ApiKeyInput => {
                            app.input.pop();
                            terminal.draw(|f| ui(f, &app))?;
                        }
                        _ => {}
                    },
                    // Handle scrolling in chat mode
                    KeyCode::PageUp => {
                        if let AppState::Chat = app.state {
                            app.scroll_up(5); // Scroll up 5 lines
                            terminal.draw(|f| ui(f, &app))?;
                        }
                    }
                    KeyCode::PageDown => {
                        if let AppState::Chat = app.state {
                            app.scroll_down(5); // Scroll down 5 lines
                            terminal.draw(|f| ui(f, &app))?;
                        }
                    }
                    // Home key for top
                    KeyCode::Home => {
                        if let AppState::Chat = app.state {
                            app.scroll_position = 0; // Scroll to top
                            terminal.draw(|f| ui(f, &app))?;
                        }
                    }
                    // End key for bottom
                    KeyCode::End => {
                        if let AppState::Chat = app.state {
                            app.auto_scroll_to_bottom(); // Scroll to bottom
                            terminal.draw(|f| ui(f, &app))?;
                        }
                    }
                    _ => {}
                }
            }
        } else {
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    Ok(())
}

fn process_message(app: &mut App, msg: &str) -> Result<()> {
    app.messages
        .push(format!("DEBUG: Processing message: {}", msg));

    if msg.starts_with("progress:") {
        // Make sure download_active is true whenever we receive progress
        app.download_active = true;

        let parts: Vec<&str> = msg.split(':').collect();
        if parts.len() >= 3 {
            if let (Ok(downloaded), Ok(total)) = (parts[1].parse::<u64>(), parts[2].parse::<u64>())
            {
                app.download_progress = Some((downloaded, total));
                // Only log progress occasionally to avoid flooding logs
                if downloaded % (5 * 1024 * 1024) < 100000 {
                    // Log roughly every 5MB
                    app.messages.push(format!(
                        "DEBUG: Download progress: {:.1}MB/{:.1}MB",
                        downloaded as f64 / 1_000_000.0,
                        total as f64 / 1_000_000.0
                    ));
                }
            }
        }
    } else if msg.starts_with("status:") {
        // Status updates for the download process
        let status = msg.replacen("status:", "", 1);
        app.messages.push(format!("Status: {}", status));
    } else if msg.starts_with("download_started:") {
        app.download_active = true;
        let url = msg.replacen("download_started:", "", 1);
        app.messages.push(format!("Starting download from {}", url));
    } else if msg == "download_complete" {
        app.download_active = false;
        app.messages
            .push("Download completed! Loading model...".into());
        let model_path = app.model_path(&app.current_model().file_name)?;
        match app.load_model(&model_path) {
            Ok(()) => {
                app.state = AppState::Chat;
                app.messages.push("Setup complete. Ready to chat!".into());
                app.messages
                    .push("You can now ask questions about coding and development.".into());
            }
            Err(e) => {
                app.messages
                    .push(format!("ERROR: Failed to load model: {}", e));
                app.state = AppState::Error(format!("Failed to load model: {}", e));
            }
        }
    } else if msg == "api_key_needed" {
        // Special case for when we need an API key
        app.messages
            .push("Please enter your Anthropic API key to use Claude 3.7...".into());
    } else if msg == "setup_complete" {
        app.state = AppState::Chat;
        app.messages.push("Setup complete. Ready to chat!".into());

        // Add the welcome message and help info
        app.messages.push("".into());
        app.messages.push("Welcome to OLI assistant!".into());
        app.messages.push("/help for help".into());
        if let Some(cwd) = &app.current_working_dir {
            app.messages.push(format!("cwd: {}", cwd));
        }
        app.messages.push("".into());
    } else if msg == "setup_failed" {
        app.messages
            .push("Setup failed. Check error messages above.".into());
    } else if msg.starts_with("error:") {
        let error_msg = msg.replacen("error:", "", 1);
        app.error_message = Some(error_msg.clone());
        app.state = AppState::Error(error_msg);
    } else if msg.starts_with("retry:") {
        app.messages.push(msg.replacen("retry:", "", 1));
    } else if msg.starts_with("Executing tool") || msg.starts_with("Running tool") {
        // Handle agent tool execution messages
        app.messages.push(format!("🔧 {}", msg));
    } else if msg.starts_with("Sending request to AI") || msg.starts_with("Processing tool results")
    {
        // Handle agent progress messages
        app.messages.push(format!("⏳ {}", msg));
    } else if msg == "Agent initialized successfully" {
        app.messages
            .push("🚀 Agent initialized and ready to use!".into());
    } else if msg.starts_with("Failed to initialize agent") {
        app.messages.push(format!("❌ {}", msg));
        app.use_agent = false;
    }

    Ok(())
}

fn initialize_messages(app: &mut App) {
    app.messages.extend(vec![
        "★ Welcome to OLI Assistant! ★".into(),
        "A terminal-based code assistant powered by local LLMs".into(),
        "".into(),
        "1. Select a model using Up/Down arrow keys".into(),
        "2. Press Enter to download and set up the selected model".into(),
        "3. After setup, you can chat with the assistant about code".into(),
        "".into(),
    ]);
}

fn ui(f: &mut Frame, app: &App) {
    match app.state {
        AppState::Setup => draw_setup(f, app),
        AppState::ApiKeyInput => draw_api_key_input(f, app),
        AppState::Chat => draw_chat(f, app),
        AppState::Error(ref error_msg) => draw_error(f, app, error_msg),
    }
}

fn draw_setup(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(2)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(4),
            Constraint::Length(3),
        ])
        .split(f.area());

    let title = Paragraph::new("OLI Setup Assistant")
        .style(
            Style::default()
                .fg(Color::LightCyan)
                .add_modifier(Modifier::BOLD),
        )
        .alignment(Alignment::Center);
    f.render_widget(title, chunks[0]);

    // Model list
    let models: Vec<ListItem> = app
        .available_models
        .iter()
        .enumerate()
        .map(|(i, model)| {
            let content = format!(
                "{} ({:.2}GB) - {}",
                model.name, model.size_gb, model.description
            );
            if i == app.selected_model {
                ListItem::new(format!("→ {}", content)).style(Style::default().fg(Color::Yellow))
            } else {
                ListItem::new(format!("  {}", content))
            }
        })
        .collect();

    let models_list = List::new(models)
        .block(Block::default().borders(Borders::ALL).title("Models"))
        .highlight_style(Style::default().fg(Color::Yellow));
    f.render_widget(models_list, chunks[1]);

    // Progress
    let progress_text = if app.download_active {
        app.download_progress.map_or_else(
            || "Preparing download...".into(),
            |(d, t)| {
                let percent = if t > 0 {
                    (d as f64 / t as f64) * 100.0
                } else {
                    0.0
                };

                // Create a visual progress bar
                let bar_width = 50; // Number of characters for the progress bar
                let filled = (percent / 100.0 * bar_width as f64) as usize;
                let empty = bar_width - filled;
                let progress_bar = format!(
                    "[{}{}] {:.1}%",
                    "=".repeat(filled),
                    " ".repeat(empty),
                    percent
                );

                format!(
                    "{}\nDownloading {}: {:.2}MB of {:.2}MB",
                    progress_bar,
                    app.current_model().file_name,
                    d as f64 / 1_000_000.0,
                    t as f64 / 1_000_000.0
                )
            },
        )
    } else {
        "Press Enter to begin setup".into()
    };

    let progress_bar = Paragraph::new(progress_text)
        .block(Block::default().borders(Borders::ALL).title("Progress"))
        .style(Style::default().fg(Color::Green));
    f.render_widget(progress_bar, chunks[2]);
}

fn draw_chat(f: &mut Frame, app: &App) {
    // Use three chunks - header, message history, and input
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(1), // Status bar
            Constraint::Min(3),    // Chat history (expandable)
            Constraint::Length(3), // Input area (fixed height)
        ])
        .split(f.area());

    // Status bar showing model info and scroll position
    let model_name = app.current_model().name.clone();
    let scroll_info = format!(
        "Scroll: {}/{} (PageUp/PageDown to scroll)",
        app.scroll_position,
        app.messages.len().saturating_sub(10)
    );

    // Add agent indicator if agent is available
    let agent_indicator = if app.use_agent && app.agent.is_some() {
        Span::styled(
            " 🤖 Agent ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::LightGreen)
                .add_modifier(Modifier::BOLD),
        )
    } else {
        Span::styled(
            " 🖥️ Local ",
            Style::default().fg(Color::Black).bg(Color::Yellow),
        )
    };

    let status_bar = Line::from(vec![
        Span::styled(
            format!(" Model: {} ", model_name),
            Style::default().fg(Color::LightCyan).bg(Color::DarkGray),
        ),
        Span::raw(" "),
        agent_indicator,
        Span::raw(" | "),
        Span::styled(scroll_info, Style::default().fg(Color::DarkGray)),
        Span::raw(" | "),
        Span::styled(
            " PgUp/PgDn: Scroll  Esc: Quit ",
            Style::default().fg(Color::Black).bg(Color::LightBlue),
        ),
    ]);

    let status_bar_widget = Paragraph::new(status_bar).style(Style::default());
    f.render_widget(status_bar_widget, chunks[0]);

    // Filter and style messages
    let visible_messages: Vec<Line> = app
        .messages
        .iter()
        .enumerate()
        // Apply scrolling - show messages based on scroll position
        .filter(|(idx, _)| {
            // Only show messages at or after the scroll position
            *idx >= app.scroll_position &&
            // Only show messages that would fit in the visible area
            *idx < app.scroll_position + chunks[1].height as usize
        })
        .map(|(_, m)| {
            if m.starts_with("DEBUG:") {
                // Only show debug messages in debug mode
                if app.debug_messages {
                    Line::from(vec![Span::styled(
                        m.as_str(),
                        Style::default().fg(Color::Yellow),
                    )])
                } else {
                    Line::from("")
                }
            } else if let Some(stripped) = m.strip_prefix("> ") {
                // User messages - cyan
                Line::from(vec![
                    Span::styled(
                        "YOU: ",
                        Style::default()
                            .fg(Color::LightBlue)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(stripped, Style::default().fg(Color::Cyan)),
                ])
            } else if m.starts_with("Error:") || m.starts_with("ERROR:") {
                // Error messages - red
                Line::from(vec![Span::styled(
                    m.as_str(),
                    Style::default().fg(Color::Red),
                )])
            } else if m.starts_with("Status:") {
                // Status messages - blue
                Line::from(vec![Span::styled(
                    m.as_str(),
                    Style::default().fg(Color::Blue),
                )])
            } else if m.starts_with("★") {
                // Title/welcome messages - light cyan with bold
                Line::from(vec![Span::styled(
                    m.as_str(),
                    Style::default()
                        .fg(Color::LightCyan)
                        .add_modifier(Modifier::BOLD),
                )])
            } else if m == "Thinking..." {
                // Thinking message
                Line::from(vec![Span::styled(
                    "🤔 Thinking...",
                    Style::default()
                        .fg(Color::LightYellow)
                        .add_modifier(Modifier::ITALIC),
                )])
            } else {
                // Model responses - with styling for code blocks
                if m.trim().is_empty() {
                    Line::from("")
                } else if !m.starts_with("> ")
                    && !m.starts_with("DEBUG:")
                    && app.messages.contains(&format!("> {}", app.input))
                {
                    // This is likely a model response
                    Line::from(vec![
                        Span::styled(
                            "OLI: ",
                            Style::default()
                                .fg(Color::LightGreen)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::raw(m),
                    ])
                } else {
                    // Regular text
                    Line::from(m.as_str())
                }
            }
        })
        .collect();

    // Create a scrollable paragraph for the messages
    let has_more_above = app.scroll_position > 0;
    let has_more_below = app.scroll_position + (chunks[1].height as usize) < app.messages.len();

    let mut message_block = Block::default()
        .borders(Borders::ALL)
        .title("OLI Assistant");

    if has_more_above {
        message_block = message_block.title(Line::from(vec![
            Span::raw("OLI Assistant "),
            Span::styled("▲ more above", Style::default().fg(Color::DarkGray)),
        ]));
    }

    if has_more_below {
        message_block = message_block.title(Line::from(vec![
            Span::raw("OLI Assistant "),
            Span::styled("▼ more below", Style::default().fg(Color::DarkGray)),
        ]));
    }

    let messages_window = Paragraph::new(Text::from(visible_messages))
        .block(message_block)
        .wrap(Wrap { trim: true });
    f.render_widget(messages_window, chunks[1]);

    // Input box with hint text
    let input_text = if app.input.is_empty() {
        Span::styled(
            "Type your code question and press Enter...",
            Style::default().fg(Color::DarkGray),
        )
    } else {
        Span::raw(app.input.as_str())
    };

    let input_window = Paragraph::new(input_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Input (Esc to quit)")
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .wrap(Wrap { trim: true });
    f.render_widget(input_window, chunks[2]);

    // Only show cursor if there is input
    if !app.input.is_empty() {
        // Set cursor position at end of input
        f.set_cursor_position((chunks[2].x + app.input.width() as u16 + 1, chunks[2].y + 1));
    }
}

fn draw_api_key_input(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(2)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(4),
            Constraint::Length(3),
        ])
        .split(f.area());

    let title = Paragraph::new("Anthropic API Key Setup")
        .style(
            Style::default()
                .fg(Color::LightCyan)
                .add_modifier(Modifier::BOLD),
        )
        .alignment(Alignment::Center);
    f.render_widget(title, chunks[0]);

    // Messages area showing info about API key requirements
    let message_items = vec![
        ListItem::new("To use Claude 3.7, you need to provide your Anthropic API key."),
        ListItem::new("You can get an API key from https://console.anthropic.com/"),
        ListItem::new(""),
        ListItem::new(
            "The API key will be used only for this session and will not be stored permanently.",
        ),
        ListItem::new(
            "You can also set the ANTHROPIC_API_KEY environment variable to avoid this prompt.",
        ),
    ];

    let messages = List::new(message_items)
        .block(Block::default().borders(Borders::ALL).title("Information"))
        .style(Style::default().fg(Color::Yellow));
    f.render_widget(messages, chunks[1]);

    // Input box with masked input for security
    let input_content = if app.input.is_empty() {
        Span::styled(
            "Enter your Anthropic API key and press Enter...",
            Style::default().fg(Color::DarkGray),
        )
    } else {
        // Mask the API key with asterisks for privacy
        Span::raw("*".repeat(app.input.len()))
    };

    let input_box = Paragraph::new(input_content)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("API Key")
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .style(Style::default())
        .alignment(Alignment::Left);
    f.render_widget(input_box, chunks[2]);

    // Set cursor position for input
    if !app.input.is_empty() {
        // Position the cursor at the end of the masked input
        f.set_cursor_position((chunks[2].x + app.input.len() as u16 + 1, chunks[2].y + 1));
    } else {
        // Position at the start of the input area
        f.set_cursor_position((chunks[2].x + 1, chunks[2].y + 1));
    }
}

fn draw_error(f: &mut Frame, _app: &App, error_msg: &str) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(2)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(4),
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
                .title("Error Details"),
        )
        .style(Style::default().fg(Color::Red))
        .wrap(Wrap { trim: true });
    f.render_widget(error_text, chunks[1]);

    let instruction = Paragraph::new("Press Enter to return to setup or Esc to exit")
        .block(Block::default().borders(Borders::ALL))
        .style(Style::default().fg(Color::Yellow))
        .alignment(Alignment::Center);
    f.render_widget(instruction, chunks[2]);
}
