use lazy_static::lazy_static;
use oli_server::agent::core::{Agent, LLMProvider};
use oli_server::app::logger::{format_log_with_color, LogLevel};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use tempfile::TempDir;

/// Structs for tool benchmark dataset
#[derive(Debug, Serialize, Deserialize)]
pub struct ToolBenchmarkParams {
    #[serde(flatten)]
    pub params: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolBenchmarkQuery {
    pub query: String,
    pub expected_tool: String,
    pub expected_params: ToolBenchmarkParams,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolBenchmarkDataset {
    pub queries: Vec<ToolBenchmarkQuery>,
}

/// Helper function to set up an agent with specified provider, or the default from environment
pub async fn setup_agent() -> Option<(Agent, u64)> {
    // Load environment variables from .env file if it exists
    let _ = dotenv::dotenv();

    // Skip if SKIP_INTEGRATION is set (useful for CI/CD environments)
    if std::env::var("SKIP_INTEGRATION").is_ok() {
        return None;
    }

    // Get provider from environment or default to Ollama
    let provider_str = env::var("TEST_PROVIDER")
        .unwrap_or_else(|_| "ollama".to_string())
        .to_lowercase();
    println!("Using provider: {}", provider_str);

    // Set up provider-specific environment
    let provider = match provider_str.as_str() {
        "anthropic" => {
            // Check for API key
            if env::var("ANTHROPIC_API_KEY").is_err() {
                println!(
                    "ANTHROPIC_API_KEY environment variable must be set for Anthropic provider"
                );
                return None;
            }
            LLMProvider::Anthropic
        }
        "openai" => {
            // Check for API key
            if env::var("OPENAI_API_KEY").is_err() {
                println!("OPENAI_API_KEY environment variable must be set for OpenAI provider");
                return None;
            }
            LLMProvider::OpenAI
        }
        "gemini" => {
            // Check for API key
            if env::var("GEMINI_API_KEY").is_err() {
                println!("GEMINI_API_KEY environment variable must be set for Gemini provider");
                return None;
            }
            LLMProvider::Gemini
        }
        _ => {
            // Default to Ollama
            // Setup needed environment for Ollama connection
            if env::var("OLLAMA_API_BASE").is_err() {
                env::set_var("OLLAMA_API_BASE", "http://localhost:11434");
            }
            LLMProvider::Ollama
        }
    };

    // Get the model from env
    let model = match env::var("TEST_MODEL") {
        Ok(m) => m,
        Err(_) => {
            // Default models based on provider
            match provider {
                LLMProvider::Anthropic => "claude-3-opus-20240229".to_string(),
                LLMProvider::OpenAI => "gpt-4-turbo".to_string(),
                LLMProvider::Gemini => "gemini-1.5-pro".to_string(),
                LLMProvider::Ollama => {
                    match env::var("DEFAULT_MODEL") {
                        Ok(m) => m,
                        Err(_) => {
                            println!("DEFAULT_MODEL environment variable must be set for Ollama provider");
                            return None;
                        }
                    }
                }
            }
        }
    };

    // Initialize the agent based on provider and model
    println!(
        "Initializing agent with provider: {}, model: {}",
        provider_str, model
    );
    let mut agent = Agent::new(provider.clone()).with_model(model);

    // For API-based providers, we need to initialize with API key
    let result = match provider {
        LLMProvider::Anthropic => {
            let api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY should be set");
            agent.initialize_with_api_key(api_key).await
        }
        LLMProvider::OpenAI => {
            let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY should be set");
            agent.initialize_with_api_key(api_key).await
        }
        LLMProvider::Gemini => {
            let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY should be set");
            agent.initialize_with_api_key(api_key).await
        }
        LLMProvider::Ollama => agent.initialize().await,
    };

    if let Err(e) = result {
        println!("Failed to initialize agent: {}", e);
        return None;
    }

    // Set a reasonable timeout
    let timeout_secs = env::var("OLI_TEST_TIMEOUT")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(120); // 2 minute default timeout

    // Return the initialized agent
    Some((agent, timeout_secs))
}

/// Helper function to set up a test directory structure with sample files for benchmarking
pub fn setup_test_files(temp_dir: &TempDir) -> std::path::PathBuf {
    let test_dir = temp_dir.path().to_path_buf();

    // Create key directories
    let dirs = [
        "src",
        "src/tools",
        "tests",
        "app/src/components",
        "docs",
        "config",
        "features/new_feature",
    ];

    for dir in dirs {
        fs::create_dir_all(test_dir.join(dir))
            .unwrap_or_else(|_| panic!("Failed to create {} directory", dir));
    }

    // Create sample files needed for the benchmark tests
    let files = [
        ("README.md", "# Sample Project\n\n## Overview\nThis is a sample project for testing.\n\n## Installation\nTo instll this project, run `cargo build`.\n\n## Usage\nDescribe how to use the project here."),
        ("LICENSE", "MIT License\n\nCopyright (c) 2022 Test Project\n\nPermission is hereby granted, free of charge..."),
        ("Cargo.toml", "[package]\nname = \"test-project\"\nversion = \"0.1.0\"\n\n[dependencies]\ntoken = \"0.1.0\"\nrequests = \"0.2.0\"\n\n[dev-dependencies]\nmockall = \"0.10.0\""),
        ("src/main.rs", "fn main() {\n    println!(\"Hello, world!\");\n}\n\n// client.send_request(data)"),
        ("src/lib.rs", "pub mod utils;\npub mod tools;\n\npub fn hello() -> &'static str {\n    \"Hello, world!\"\n}\n\nfn internal_func() {\n    internal_func()\n}"),
        ("src/tools/mod.rs", "pub mod file;\npub mod search;\n\npub fn execute() {\n    println!(\"Executing...\");\n}"),
        ("src/utils.rs", "pub fn read_file(path: &str) -> Result<String, std::io::Error> {\n    std::fs::read_to_string(path)\n}\n\nasync fn fetch_data() -> Result<String, Box<dyn std::error::Error>> {\n    Ok(\"data\".to_string())\n}"),
        ("tests/test_main.rs", "// TODO: Add more comprehensive tests\n#[test]\nfn test_hello() {\n    assert_eq!(test_project::hello(), \"Hello, world!\");\n}"),
        ("config.json", "{\n  \"debug\": false,\n  \"port\": 8080,\n  \"api_key\": \"test_key\"\n}"),
        ("config.js", "module.exports = {\n  database: 'mongodb://localhost:27017/app',\n  logLevel: 'info'\n};"),
        ("logger.conf", "# Logger configuration\noutput = stdout\nlevel = INFO\nformat = {level}: {message}"),
        ("package.json", "{\n  \"name\": \"app\",\n  \"version\": \"1.0.0\",\n  \"description\": \"An application\",\n  \"main\": \"index.js\",\n  \"scripts\": {\n    \"test\": \"echo \\\"Error: no test specified\\\" && exit 1\",\n    \"start\": \"node index.js\"\n  },\n  \"author\": \"\",\n  \"license\": \"MIT\"\n}"),
        ("app/src/components/Button.tsx", "import React, { useState, useEffect } from 'react';\n\ninterface ButtonProps {\n  text: string;\n  onClick: () => void;\n}\n\nexport default function Button({ text, onClick }: ButtonProps) {\n  const [clicked, setClicked] = useState(false);\n  \n  useEffect(() => {\n    // Reset click state after 1 second\n    if (clicked) {\n      const timer = setTimeout(() => setClicked(false), 1000);\n      return () => clearTimeout(timer);\n    }\n  }, [clicked]);\n  \n  return (\n    <button onClick={() => {\n      setClicked(true);\n      onClick();\n    }}>\n      {text}\n    </button>\n  );\n}"),
        ("test.txt", "This is a test file with a bug that needs to be fixed."),
        ("CONTRIBUTING.md", "# Contributing Guide\n\nThank you for considering contributing to this project!\n\n## Code of Conduct\n\nPlease follow our Code of Conduct.\n\n## Questions\n\nIf you have questions, please email contact@example.com.")
    ];

    for (filename, content) in files {
        let file_path = test_dir.join(filename);
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)
                .unwrap_or_else(|_| panic!("Failed to create parent directory for {}", filename));
        }
        fs::write(file_path, content).unwrap_or_else(|_| panic!("Failed to write {}", filename));
    }

    test_dir
}

/// Helper function to compare expected and actual tool call parameters
/// Focus only on verifying that the correct tool was called with proper parameters
pub fn compare_tool_params(
    expected_tool: &str,
    expected_params: &ToolBenchmarkParams,
    actual_tool: &str,
    actual_params: &serde_json::Value,
    test_dir: &Path,
) -> bool {
    // First check if we have a tool call
    if actual_tool.is_empty() {
        println!("No tool call detected");
        return false;
    }

    // Check if the tool names match
    if expected_tool != actual_tool {
        println!(
            "Tool name mismatch: expected {}, got {}",
            expected_tool, actual_tool
        );
        return false;
    }

    // Now check all parameters
    let mut all_params_match = true;

    // For each expected parameter, check if it exists and has the correct value
    for (key, expected_value) in &expected_params.params {
        // Check if the parameter exists in actual params
        if let Some(actual_value) = actual_params.get(key) {
            // Special handling for paths with {TEST_DIR} placeholder
            let expected_value_normalized = if let Some(expected_str) = expected_value.as_str() {
                if expected_str.contains("{TEST_DIR}") {
                    let test_dir_str = test_dir.to_string_lossy();
                    serde_json::Value::String(expected_str.replace("{TEST_DIR}", &test_dir_str))
                } else {
                    expected_value.clone()
                }
            } else {
                expected_value.clone()
            };

            // Compare the normalized expected value with the actual value
            if expected_value_normalized != *actual_value {
                println!(
                    "Parameter '{}' value mismatch: expected {:?}, got {:?}",
                    key, expected_value_normalized, actual_value
                );
                all_params_match = false;
            }
        } else {
            // Parameter is missing entirely
            println!("Missing parameter '{}' in actual params", key);
            all_params_match = false;
        }
    }

    all_params_match
}

/// Initialize logging for tests
pub fn init_logging() {
    // Create logs directory if it doesn't exist
    let log_dir = Path::new("logs");
    if !log_dir.exists() {
        fs::create_dir_all(log_dir).expect("Failed to create logs directory");
    }

    // Print starting message to both stdout and stderr to test which one appears
    println!("\n==== STARTING TEST ====");
    eprintln!("\n==== STARTING TEST ====");

    // Also write directly to both streams
    let _ = io::stdout().write_all(b"\nSTDOUT TEST MESSAGE\n");
    let _ = io::stdout().flush();
    let _ = io::stderr().write_all(b"\nSTDERR TEST MESSAGE\n");
    let _ = io::stderr().flush();
}

/// Helper function to log messages with timestamps and color coding
pub fn log(level: LogLevel, message: &str) {
    let formatted = format_log_with_color(level, message);

    // Use eprintln to write to stderr
    eprintln!("{}", formatted);

    // Also write directly to stderr to ensure it's not captured
    let _ = io::stderr().write_all(format!("{}\n", formatted).as_bytes());
    let _ = io::stderr().flush();

    // Additionally, write to a logfile for persistence
    // Use a static variable to store the log filename so we only create it once per test run
    lazy_static! {
        static ref LOG_FILE: String = format!(
            "logs/test_{}.log",
            chrono::Local::now().format("%Y%m%d_%H%M%S")
        );
    }

    // Ensure the logs directory exists
    let log_path = Path::new(&*LOG_FILE);
    if let Some(parent) = log_path.parent() {
        if !parent.exists() {
            let _ = fs::create_dir_all(parent);
        }
    }

    // Write to the log file
    if let Ok(mut file) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(LOG_FILE.as_str())
    {
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
        let _ = writeln!(file, "[{}] [{}] {}", timestamp, level.as_str(), message);
        let _ = file.flush(); // Ensure it's written immediately
    }

    // For critical messages, also write to a shared logfile that's always in the same location
    if level == LogLevel::Error || level == LogLevel::Warning {
        if let Ok(mut file) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("logs/latest.log")
        {
            let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
            let _ = writeln!(file, "[{}] [{}] {}", timestamp, level.as_str(), message);
            let _ = file.flush();
        }
    }
}
