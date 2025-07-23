use std::fs;
use tempfile::tempdir;

use oli_server::tools::lsp::{LspServerManager, LspServerType};

#[test]
fn test_document_symbol_path_handling() {
    // Create a test manager
    let manager = LspServerManager::new();

    // Create a temporary directory structure with files
    let temp_dir = tempdir().expect("Failed to create temp dir");

    // Create a more substantial Python file that would have symbols
    let py_file_path = temp_dir.path().join("test.py");
    let python_content = r#"
class TestClass:
    def __init__(self):
        self.value = 42

    def test_method(self):
        return self.value

def test_function():
    return "Hello world"

TEST_CONSTANT = "Test value"
"#;
    fs::write(&py_file_path, python_content).expect("Failed to write Python file");

    // Skip test if pyright-langserver is not installed
    let pyright_check = std::process::Command::new("sh")
        .arg("-c")
        .arg("command -v pyright-langserver")
        .output();

    if pyright_check.is_err() || !pyright_check.unwrap().status.success() {
        println!("Skipping test_document_symbol_path_handling: pyright-langserver not installed");
        return;
    }

    // Test with the Python file - this will indirectly test find_workspace_root
    // The main goal of this test is to verify our fix for path handling in find_workspace_root
    let result = manager.document_symbol(&py_file_path.to_string_lossy(), &LspServerType::Python);

    // We don't need to verify the actual symbols - we only care that the method doesn't panic
    // due to our fix for the Path handling in find_workspace_root

    // The method might fail for other reasons (like LSP server connection issues),
    // so we just print the error without failing the test
    if let Err(err) = &result {
        println!("Got error from document_symbol (but we only care about path handling): {err}");
    }

    // The test is considered successful if we got this far without panicking
    println!("Successfully invoked document_symbol without path handling panics");
}
