use oli_server::tools::fs::file_ops::FileOps;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use tempfile::tempdir;

// Helper function to create a test file with content
fn create_test_file(dir: &Path, filename: &str, content: &str) -> std::path::PathBuf {
    let file_path = dir.join(filename);
    let mut file = File::create(&file_path).unwrap();
    writeln!(file, "{content}").unwrap();
    file_path
}

#[test]
fn test_read_file() {
    let dir = tempdir().unwrap();
    let content = "This is a test file\nWith multiple lines\nFor testing file operations";
    let file_path = create_test_file(dir.path(), "test.txt", content);

    let result = FileOps::read_file(&file_path).unwrap();
    assert_eq!(result, format!("{content}\n"));
}

#[test]
fn test_read_file_with_line_numbers() {
    let dir = tempdir().unwrap();
    let content = "This is a test file\nWith multiple lines\nFor testing file operations";
    let file_path = create_test_file(dir.path(), "test.txt", content);

    let result = FileOps::read_file_with_line_numbers(&file_path).unwrap();
    let expected = "   1 | This is a test file\n   2 | With multiple lines\n   3 | For testing file operations";
    assert_eq!(result, expected);
}

#[test]
fn test_read_file_lines_with_offset_and_limit() {
    let dir = tempdir().unwrap();
    let content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5";
    let file_path = create_test_file(dir.path(), "test.txt", content);

    // Test with offset 1 and limit 2
    let result = FileOps::read_file_lines(&file_path, 1, Some(2)).unwrap();
    let expected = "   2 | Line 2\n   3 | Line 3";
    assert_eq!(result, expected);

    // Test with offset 2 and no limit
    let result = FileOps::read_file_lines(&file_path, 2, None).unwrap();
    let expected = "   3 | Line 3\n   4 | Line 4\n   5 | Line 5";
    assert_eq!(result, expected);

    // Test with offset beyond file length
    let result = FileOps::read_file_lines(&file_path, 10, Some(2)).unwrap();
    let expected = "";
    assert_eq!(result, expected);

    // Test with offset 0 and limit beyond file length
    let result = FileOps::read_file_lines(&file_path, 0, Some(10)).unwrap();
    let expected = "   1 | Line 1\n   2 | Line 2\n   3 | Line 3\n   4 | Line 4\n   5 | Line 5";
    assert_eq!(result, expected);
}

#[test]
fn test_write_file() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("new_file.txt");
    let content = "This is new content to write";

    // Write content to file
    FileOps::write_file(&file_path, content).unwrap();

    // Read back the content to verify
    let result = FileOps::read_file(&file_path).unwrap();
    assert_eq!(result, content);
}

#[test]
fn test_edit_file() {
    let dir = tempdir().unwrap();
    let content = "This is the original content\nThat needs to be modified\nFor testing purposes";
    let file_path = create_test_file(dir.path(), "edit.txt", content);

    // Edit the file by replacing a string
    let old_string = "original content";
    let new_string = "modified content";
    FileOps::edit_file(&file_path, old_string, new_string, None).unwrap();

    // Read back the content to verify
    let result = FileOps::read_file(&file_path).unwrap();
    let expected =
        "This is the modified content\nThat needs to be modified\nFor testing purposes\n";
    assert_eq!(result, expected);
}

#[test]
fn test_list_directory() {
    let dir = tempdir().unwrap();

    // Create some test files
    create_test_file(dir.path(), "file1.txt", "Content 1");
    create_test_file(dir.path(), "file2.txt", "Content 2");

    let result = FileOps::list_directory(dir.path()).unwrap();
    assert_eq!(result.len(), 2);

    // Sort by name so the order is predictable for testing
    let file_names: Vec<String> = result
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
        .collect();

    assert!(file_names.contains(&"file1.txt".to_string()));
    assert!(file_names.contains(&"file2.txt".to_string()));
}

// This test specifically tests the logging of offset and limit
#[test]
fn test_read_file_lines_edge_cases() {
    let dir = tempdir().unwrap();
    let content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5";
    let file_path = create_test_file(dir.path(), "test.txt", content);

    // Test empty file
    let empty_file = create_test_file(dir.path(), "empty.txt", "");
    // The file actually has a newline because of the create_test_file function
    let result = FileOps::read_file_lines(&empty_file, 0, None).unwrap();
    // So it has one empty line with a line number
    assert_eq!(result, "   1 | ");

    // Test offset at file boundary
    let result = FileOps::read_file_lines(&file_path, 5, None).unwrap();
    assert_eq!(result, "");

    // Test zero limit
    let result = FileOps::read_file_lines(&file_path, 0, Some(0)).unwrap();
    assert_eq!(result, "");

    // Test large offset and large limit
    let result = FileOps::read_file_lines(&file_path, 100, Some(100)).unwrap();
    assert_eq!(result, "");
}

// Test file operations with non-existent files
#[test]
fn test_file_operations_errors() {
    let non_existent_path = Path::new("/non/existent/file.txt");

    // Test read operations
    assert!(FileOps::read_file(non_existent_path).is_err());
    assert!(FileOps::read_file_with_line_numbers(non_existent_path).is_err());
    assert!(FileOps::read_file_lines(non_existent_path, 0, None).is_err());

    // Test list directory
    assert!(FileOps::list_directory(non_existent_path).is_err());
}

// Test edit file with multiple occurrences
#[test]
fn test_edit_file_with_multiple_occurrences() {
    let dir = tempdir().unwrap();
    let content = "This pattern appears twice\nThis pattern appears twice\nEnd of file";
    let file_path = create_test_file(dir.path(), "duplicate.txt", content);

    // This should fail because the pattern occurs multiple times
    let result = FileOps::edit_file(
        &file_path,
        "This pattern appears twice",
        "Replacement",
        None,
    );
    assert!(result.is_err());

    // The error message should mention multiple occurrences
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("multiple times"));
}

// Test edit file with non-existent pattern
#[test]
fn test_edit_file_with_non_existent_pattern() {
    let dir = tempdir().unwrap();
    let content = "This is a test file\nFor testing edit operations";
    let file_path = create_test_file(dir.path(), "edit_test.txt", content);

    // This should fail because the pattern doesn't exist
    let result = FileOps::edit_file(&file_path, "non-existent pattern", "Replacement", None);
    assert!(result.is_err());

    // The error message should indicate that the pattern wasn't found
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("not found"));
}

// Test edit file with expected_replacements parameter
#[test]
fn test_edit_file_with_expected_replacements() {
    let dir = tempdir().unwrap();
    let content = "Repeat this.\nRepeat this.\nRepeat this.\nEnd of file.";
    let file_path = create_test_file(dir.path(), "expected_replacements.txt", content);

    // This should work because we specified the exact number of occurrences
    let result = FileOps::edit_file(&file_path, "Repeat this.", "Changed line.", Some(3));
    assert!(result.is_ok());

    // Read back the content to verify all occurrences were replaced
    let result = FileOps::read_file(&file_path).unwrap();
    let expected = "Changed line.\nChanged line.\nChanged line.\nEnd of file.\n";
    assert_eq!(result, expected);

    // Test with incorrect expected count
    let incorrect_file_path = create_test_file(dir.path(), "wrong_count.txt", content);
    let result = FileOps::edit_file(
        &incorrect_file_path,
        "Repeat this.",
        "Changed line.",
        Some(2), // There are actually 3 occurrences
    );
    assert!(result.is_err());

    // The error message should mention the mismatch in counts
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Found 3 occurrences") && err_msg.contains("expected exactly 2"));
}
