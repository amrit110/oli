use super::core::App;
use anyhow::Result;

impl App {
    /// Read the memory file content
    pub fn read_memory(&self) -> Result<String> {
        self.memory_manager.read_memory()
    }

    /// Write content to the memory file
    pub fn write_memory(&self, content: &str) -> Result<()> {
        self.memory_manager.write_memory(content)
    }

    /// Add a memory entry to a specific section
    pub fn add_memory(&self, section: &str, memory: &str) -> Result<()> {
        self.memory_manager.add_memory(section, memory)
    }

    /// Get all memories in structured form
    pub fn get_memories(&self) -> Result<Vec<(String, Vec<String>)>> {
        self.memory_manager.parse_memory()
    }

    /// Get the memory file path as a string
    pub fn memory_path(&self) -> String {
        self.memory_manager.memory_path().display().to_string()
    }
}
