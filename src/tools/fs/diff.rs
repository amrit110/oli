use anyhow::Result;
use std::fmt::Write;

/// Represents a line in a diff
#[derive(Debug)]
pub enum DiffLine {
    Added(String),
    Removed(String),
    Context(String),
}

/// Utilities for generating and formatting diffs between text
pub struct DiffTools;

impl DiffTools {
    /// Generate a diff between two strings
    pub fn generate_diff(old_text: &str, new_text: &str) -> Vec<DiffLine> {
        let old_lines: Vec<&str> = old_text.lines().collect();
        let new_lines: Vec<&str> = new_text.lines().collect();

        // Simple line-by-line diff implementation
        let mut diff = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < old_lines.len() || j < new_lines.len() {
            if i < old_lines.len() && j < new_lines.len() && old_lines[i] == new_lines[j] {
                // Line is the same
                diff.push(DiffLine::Context(old_lines[i].to_string()));
                i += 1;
                j += 1;
            } else {
                // Find next matching line
                let mut found_match = false;

                // Look ahead in new_lines
                for look_ahead in 0..3 {
                    if i < old_lines.len()
                        && j + look_ahead < new_lines.len()
                        && old_lines[i] == new_lines[j + look_ahead]
                    {
                        // Found a matching line in new_lines, add the added lines before it
                        for k in 0..look_ahead {
                            diff.push(DiffLine::Added(new_lines[j + k].to_string()));
                        }
                        j += look_ahead;
                        found_match = true;
                        break;
                    }
                }

                // Look ahead in old_lines if no match found
                if !found_match {
                    for look_ahead in 0..3 {
                        if i + look_ahead < old_lines.len()
                            && j < new_lines.len()
                            && old_lines[i + look_ahead] == new_lines[j]
                        {
                            // Found a matching line in old_lines, add the removed lines before it
                            for k in 0..look_ahead {
                                diff.push(DiffLine::Removed(old_lines[i + k].to_string()));
                            }
                            i += look_ahead;
                            found_match = true;
                            break;
                        }
                    }
                }

                // If no match found, add one line as difference
                if !found_match {
                    if i < old_lines.len() {
                        diff.push(DiffLine::Removed(old_lines[i].to_string()));
                        i += 1;
                    }
                    if j < new_lines.len() {
                        diff.push(DiffLine::Added(new_lines[j].to_string()));
                        j += 1;
                    }
                }
            }
        }

        diff
    }

    /// Format diff as a string with line numbers and colors
    pub fn format_diff(diff: &[DiffLine], file_path: &str) -> Result<String> {
        let mut output = String::new();
        let mut line_number = 0;
        let mut adds = 0;
        let mut removes = 0;

        // Count additions and removals first
        for line in diff {
            match line {
                DiffLine::Added(_) => adds += 1,
                DiffLine::Removed(_) => removes += 1,
                _ => {}
            }
        }

        // Add header
        writeln!(
            output,
            "  ⎿  Updated {} with {} addition{} and {} removal{}",
            file_path,
            adds,
            if adds == 1 { "" } else { "s" },
            removes,
            if removes == 1 { "" } else { "s" }
        )?;

        // Only show the diff if there are changes
        if adds > 0 || removes > 0 {
            // Add the diff content with line numbers and colored indicators
            for line in diff {
                match line {
                    DiffLine::Context(text) => {
                        line_number += 1;
                        writeln!(output, "     {line_number:3}  {text}")?;
                    }
                    DiffLine::Added(text) => {
                        line_number += 1;
                        // Use ANSI colors to show additions in light green
                        writeln!(output, "     \x1b[92m{line_number:3}+ {text}\x1b[0m")?;
                    }
                    DiffLine::Removed(text) => {
                        // For removed lines, use a darker red color
                        // Don't increment line number for removed lines
                        writeln!(output, "     \x1b[91m{line_number:3}- {text}\x1b[0m")?;
                    }
                }
            }
        }

        Ok(output)
    }
}
