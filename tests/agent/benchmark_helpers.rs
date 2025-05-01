use crate::agent::test_tools::ToolBenchmarkParams;
use oli_server::apis::api_client::ToolCall as ApiToolCall;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

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

/// Helper function to extract tool call from LLM agent response
pub fn extract_tool_call(response: &str) -> Option<ApiToolCall> {
    // Try to parse the response as JSON to extract tool calls
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(response) {
        if let Some(tool_calls) = parsed.get("tool_calls") {
            if let Some(tool_calls_array) = tool_calls.as_array() {
                if !tool_calls_array.is_empty() {
                    let first_call = &tool_calls_array[0];
                    let name = first_call
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();
                    let id = first_call
                        .get("id")
                        .and_then(|i| i.as_str())
                        .map(|s| s.to_string());
                    let arguments = first_call
                        .get("arguments")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);

                    return Some(ApiToolCall {
                        id,
                        name,
                        arguments,
                    });
                }
            }
        }
    }
    None
}

/// Helper function to compare expected and actual tool call parameters
pub fn compare_tool_params(
    expected_tool: &str,
    expected_params: &ToolBenchmarkParams,
    actual_tool: &str,
    actual_params: &serde_json::Value,
    test_dir: &Path,
) -> bool {
    // First check if the tool names match
    if expected_tool != actual_tool {
        println!(
            "Tool name mismatch: expected {}, got {}",
            expected_tool, actual_tool
        );
        return false;
    }

    // Extract expected params
    let mut all_params_match = true;

    for (key, expected_value) in &expected_params.params {
        // Skip path parameters for now
        if let Some(expected_str) = expected_value.as_str() {
            if expected_str.contains("{TEST_DIR}") {
                continue;
            }
        }

        // Check if the parameter exists in actual params
        if let Some(actual_value) = actual_params.get(key) {
            // Replace {TEST_DIR} placeholder if necessary
            let expected_value_str = if let Some(expected_str) = expected_value.as_str() {
                if expected_str.contains("{TEST_DIR}") {
                    let test_dir_str = test_dir.to_string_lossy();
                    serde_json::Value::String(expected_str.replace("{TEST_DIR}", &test_dir_str))
                } else {
                    expected_value.clone()
                }
            } else {
                expected_value.clone()
            };

            // Compare the values
            if expected_value_str != *actual_value {
                println!(
                    "Parameter '{}' value mismatch: expected {:?}, got {:?}",
                    key, expected_value_str, actual_value
                );
                all_params_match = false;
                break;
            }
        } else {
            println!("Missing parameter '{}' in actual params", key);
            all_params_match = false;
            break;
        }
    }

    all_params_match
}
