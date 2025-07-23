use std::error::Error;
use std::fmt;

/// Main error type for the application
#[derive(Debug)]
pub enum AppError {
    /// I/O errors from std::io operations
    IoError(std::io::Error),
    /// Network errors from API requests
    NetworkError(String),
    /// LLM errors for model-specific issues
    /// Currently not used but available for future use for model-specific errors
    #[allow(dead_code)]
    LLMError(String),
    /// File operation errors
    /// Currently not used but available for future use
    #[allow(dead_code)]
    FileError(String),
    /// Parser errors for code and content parsing issues
    /// Currently not used but available for future use
    #[allow(dead_code)]
    ParserError(String),
    /// Tool execution errors
    /// Currently not used but available for future use
    #[allow(dead_code)]
    ToolError(String),
    /// Generic errors for cases not covered by other variants
    Other(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::IoError(e) => write!(f, "IO Error: {e}"),
            AppError::NetworkError(msg) => write!(f, "Network Error: {msg}"),
            AppError::LLMError(msg) => write!(f, "Model Error: {msg}"),
            AppError::FileError(msg) => write!(f, "File Error: {msg}"),
            AppError::ParserError(msg) => write!(f, "Parser Error: {msg}"),
            AppError::ToolError(msg) => write!(f, "Tool Error: {msg}"),
            AppError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl Error for AppError {}

impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> Self {
        AppError::IoError(err)
    }
}

impl From<reqwest::Error> for AppError {
    fn from(err: reqwest::Error) -> Self {
        AppError::NetworkError(err.to_string())
    }
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        AppError::Other(err.to_string())
    }
}
