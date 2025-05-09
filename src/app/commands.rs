use serde::{Deserialize, Serialize};

/// Special command definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialCommand {
    pub name: String,
    pub description: String,
}

impl SpecialCommand {
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
        }
    }
}

/// Get available commands
pub fn get_available_commands() -> Vec<SpecialCommand> {
    vec![
        SpecialCommand::new("/help", "Show help and available commands"),
        SpecialCommand::new("/clear", "Clear conversation history"),
        SpecialCommand::new("/exit", "Exit the application"),
        SpecialCommand::new("/memory", "Display and manage codebase memory"),
    ]
}
