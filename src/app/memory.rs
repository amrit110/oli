use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Structure to manage the memory file (oli.md)
pub struct MemoryManager {
    /// Path to the oli.md memory file
    memory_file_path: PathBuf,
}

impl MemoryManager {
    /// Create a new memory manager with default path
    pub fn new() -> Self {
        // Default to creating the oli.md file in the current working directory
        let memory_file_path = PathBuf::from("oli.md");
        Self { memory_file_path }
    }

    /// Create a new memory manager with a specific path
    pub fn with_path<P: AsRef<Path>>(path: P) -> Self {
        let memory_file_path = PathBuf::from(path.as_ref());
        Self { memory_file_path }
    }

    /// Read the memory file content or create a default one if it doesn't exist
    pub fn read_memory(&self) -> Result<String> {
        if self.memory_file_path.exists() {
            fs::read_to_string(&self.memory_file_path).with_context(|| {
                format!(
                    "Failed to read memory file: {}",
                    self.memory_file_path.display()
                )
            })
        } else {
            // Return a default template if the file doesn't exist
            Ok(Self::default_memory_template())
        }
    }

    /// Write memory content to the file
    pub fn write_memory(&self, content: &str) -> Result<()> {
        fs::write(&self.memory_file_path, content).with_context(|| {
            format!(
                "Failed to write to memory file: {}",
                self.memory_file_path.display()
            )
        })
    }

    /// Check if a memory file exists
    pub fn memory_exists(&self) -> bool {
        self.memory_file_path.exists()
    }

    /// Get the memory file path
    pub fn memory_path(&self) -> &Path {
        &self.memory_file_path
    }

    /// Generate a default memory template
    pub fn default_memory_template() -> String {
        r#"# oli.md

This file provides guidance to oli when working with code in this repository.

## Project Structure
- Add memories about project structure here

## Build Commands
- Add memories about build commands here

## Test Commands
- Add memories about test commands here

## Architecture
- Add memories about architecture here
"#
        .to_string()
    }

    /// Add a new memory entry to a specific section
    pub fn add_memory(&self, section: &str, memory: &str) -> Result<()> {
        let mut content = self.read_memory()?;

        // Find the section in the content
        let section_heading = format!("## {}", section);

        if let Some(section_pos) = content.find(&section_heading) {
            // Find where to insert the new memory
            let section_end = content[section_pos..]
                .find("\n## ")
                .map(|pos| section_pos + pos)
                .unwrap_or(content.len());

            // Check if there are existing memories
            let insert_pos = section_pos + section_heading.len();
            let mut updated_content = content[..insert_pos].to_owned();

            // Add the new memory entry with a newline
            updated_content.push('\n');
            updated_content.push_str("- ");
            updated_content.push_str(memory);
            updated_content.push('\n');

            // Add the rest of the content
            updated_content.push_str(&content[insert_pos..section_end]);
            updated_content.push_str(&content[section_end..]);

            // Write the updated content back to the file
            self.write_memory(&updated_content)
        } else {
            // Section doesn't exist, add it
            content.push_str(&format!("\n## {}\n- {}\n", section, memory));
            self.write_memory(&content)
        }
    }

    /// Parse memory content into a structured form
    pub fn parse_memory(&self) -> Result<Vec<(String, Vec<String>)>> {
        let content = self.read_memory()?;
        let mut sections = Vec::new();
        let mut current_section = None;
        let mut current_memories = Vec::new();

        for line in content.lines() {
            let trimmed = line.trim();

            // Check for section headers (## Section)
            if let Some(section_name) = trimmed.strip_prefix("## ") {
                // Store previous section if there was one
                if let Some(section) = current_section.take() {
                    sections.push((section, std::mem::take(&mut current_memories)));
                }

                // Start new section
                current_section = Some(section_name.to_string());
            }
            // Check for memory entries (- Memory)
            else if let Some(memory_content) = trimmed.strip_prefix("- ") {
                if current_section.is_some() {
                    current_memories.push(memory_content.to_string());
                }
            }
        }

        // Add the last section if there is one
        if let Some(section) = current_section {
            sections.push((section, current_memories));
        }

        Ok(sections)
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}
