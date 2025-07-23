use anyhow::{Context, Result};
use glob::glob;
use ignore::WalkBuilder;
use regex::Regex;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use walkdir::{DirEntry, WalkDir};

pub struct SearchTools;

impl SearchTools {
    /// Finds the project root directory by looking for common repository marker files
    fn find_project_root(start_dir: &Path) -> Option<PathBuf> {
        let mut current = start_dir.to_path_buf();

        // Check for common project markers
        let root_markers = [
            ".git",
            ".hg",
            "Cargo.toml",
            "package.json",
            "setup.py",
            "Makefile",
        ];

        loop {
            for marker in &root_markers {
                if current.join(marker).exists() {
                    return Some(current);
                }
            }

            // Move up one directory
            match current.parent() {
                Some(parent) => current = parent.to_path_buf(),
                None => break,
            }
        }

        None
    }

    /// Checks if a repository uses ignore files by looking for .gitignore, .npmignore, etc.
    fn has_ignore_files(dir: &Path) -> bool {
        let ignore_files = [".gitignore", ".npmignore", ".dockerignore"];

        // Check current directory first
        for file in &ignore_files {
            if dir.join(file).exists() {
                return true;
            }
        }

        // Then check for ignore files in subdirectories (to a limited depth)
        let max_depth = 3;
        let mut has_ignore = false;

        let entries = WalkDir::new(dir)
            .max_depth(max_depth)
            .into_iter()
            .filter_map(|e| e.ok());

        for entry in entries {
            let filename = entry.file_name().to_string_lossy();
            if ignore_files.iter().any(|&f| f == filename) {
                has_ignore = true;
                break;
            }
        }

        has_ignore
    }

    // Check if path should be ignored based on common patterns (fallback for when ignore files aren't used)
    fn is_ignored_path(path: &Path) -> bool {
        let path_str = path.to_string_lossy();

        // Common directories to ignore
        let ignored_dirs = [
            "/target/",
            "/node_modules/",
            "/.git/",
            "/dist/",
            "/build/",
            "/.cache/",
            "/coverage/",
            "/.next/",
            "/.nuxt/",
            "/venv/",
            "/.venv/",
            "/env/",
            "/__pycache__/",
            "/out/",
            "/bin/",
            "/obj/",
        ];

        // Check if the path contains any of the ignored directories
        for dir in &ignored_dirs {
            if path_str.contains(dir) {
                return true;
            }
        }

        // Ignore common generated or binary files
        let ignored_extensions = [
            ".pyc",
            ".pyo",
            ".so",
            ".o",
            ".a",
            ".lib",
            ".dll",
            ".exe",
            ".jar",
            ".war",
            ".ear",
            ".class",
            ".min.js",
            ".min.css",
            ".map",
            ".bundle.js",
            ".swp",
            ".swo",
            ".db",
            ".sqlite",
            ".sqlite3",
            ".lock",
            ".log",
            ".tmp",
            ".temp",
            ".bak",
        ];

        // First check exact file extensions
        if let Some(extension) = path.extension() {
            let ext = format!(".{}", extension.to_string_lossy());
            if ignored_extensions.contains(&ext.as_str()) {
                return true;
            }
        }

        // Then check path suffixes for special cases like minified files
        if path_str.ends_with(".min.js")
            || path_str.ends_with(".min.css")
            || path_str.ends_with(".bundle.js")
            || path_str.ends_with(".map")
        {
            return true;
        }

        false
    }

    pub fn glob_search(pattern: &str) -> Result<Vec<PathBuf>> {
        // First, get the raw entries from glob
        let raw_entries =
            glob(pattern).with_context(|| format!("Invalid glob pattern: {pattern}"))?;

        // Collect paths that match the glob pattern
        let mut glob_matches = Vec::new();
        for entry in raw_entries {
            let path = entry.context("Failed to read glob entry")?;
            glob_matches.push(path);
        }

        // If there are no matches or only one match, no need for complex filtering
        if glob_matches.is_empty() || glob_matches.len() == 1 {
            return Ok(glob_matches);
        }

        // Try to find the project root to respect repository-specific ignore files
        let mut matches = Vec::new();

        // Find a common parent directory to use as search base
        let common_parent = if let Some(first) = glob_matches.first() {
            first.parent().map(|parent| parent.to_path_buf())
        } else {
            None
        };

        if let Some(base_dir) = common_parent.as_deref().and_then(Self::find_project_root) {
            if Self::has_ignore_files(&base_dir) {
                // Repository has ignore files - use the ignore-aware walker
                let walker = WalkBuilder::new(base_dir)
                    .hidden(false) // Don't skip hidden files by default
                    .standard_filters(true) // Use .gitignore etc.
                    .build();

                // Mark when we've finished processing to avoid redundant work
                let processed = Arc::new(AtomicBool::new(false));

                for entry in walker.flatten() {
                    let path = entry.path().to_path_buf();

                    // Only include paths that were in the original glob matches
                    if glob_matches.iter().any(|m| m == &path) {
                        matches.push(path);
                        // Mark that we've processed at least one path
                        processed.store(true, Ordering::SeqCst);
                    }
                }

                // If we processed paths through the ignore-aware walker, return those results
                if processed.load(Ordering::SeqCst) {
                    // Sort by last modified time before returning
                    matches.sort_by(|a, b| {
                        let a_modified = std::fs::metadata(a).and_then(|m| m.modified()).ok();
                        let b_modified = std::fs::metadata(b).and_then(|m| m.modified()).ok();
                        b_modified.cmp(&a_modified)
                    });

                    return Ok(matches);
                }
            }
        }

        // Fallback to the default method if repository-specific ignore patterns can't be used
        for path in glob_matches {
            // Skip paths based on common ignore patterns
            if Self::is_ignored_path(&path) {
                continue;
            }
            matches.push(path);
        }

        // Sort by last modified time (most recent first)
        matches.sort_by(|a, b| {
            let a_modified = std::fs::metadata(a).and_then(|m| m.modified()).ok();
            let b_modified = std::fs::metadata(b).and_then(|m| m.modified()).ok();
            b_modified.cmp(&a_modified)
        });

        Ok(matches)
    }

    pub fn glob_search_in_dir(dir: &Path, pattern: &str) -> Result<Vec<PathBuf>> {
        let dir_str = dir.to_string_lossy();
        let full_pattern = format!("{dir_str}/{pattern}");
        Self::glob_search(&full_pattern)
    }

    // Helper function for WalkDir to skip ignored directories
    fn should_skip_dir(entry: &DirEntry) -> bool {
        let path = entry.path();

        if entry.file_type().is_dir() {
            let file_name = entry.file_name().to_string_lossy();

            // Skip common directories that should be ignored
            let ignored_dirs = [
                "target",
                "node_modules",
                ".git",
                "dist",
                "build",
                ".cache",
                "coverage",
                ".next",
                ".nuxt",
                "venv",
                ".venv",
                "env",
                "__pycache__",
                "out",
                "bin",
                "obj",
            ];

            if ignored_dirs.contains(&file_name.as_ref()) {
                return true;
            }
        }

        // Use the general path ignoring function for files
        if entry.file_type().is_file() && Self::is_ignored_path(path) {
            return true;
        }

        false
    }

    /// Convert a file pattern to a regex that the ignore walker can use for filtering
    fn create_file_pattern_filter(include_pattern: Option<&str>) -> Option<Regex> {
        match include_pattern {
            Some(pattern) => {
                let pattern = pattern.replace("*.{", "*.").replace("}", "|*."); // Convert *.{ts,tsx} to *.ts|*.tsx
                let parts: Vec<&str> = pattern.split('|').collect();
                let regex_parts: Vec<String> = parts
                    .iter()
                    .map(|p| format!("({})", glob_to_regex(p)))
                    .collect();
                let joined = regex_parts.join("|");
                Regex::new(&joined).ok()
            }
            None => None,
        }
    }

    /// Check if a file is likely binary or generated based on extension and path
    fn is_likely_binary_or_generated(path: &Path) -> bool {
        // Check for binary file extensions
        if let Some(extension) = path.extension() {
            let ext = extension.to_string_lossy();
            let binary_extensions = [
                "pyc", "pyo", "so", "o", "a", "lib", "dll", "exe", "jar", "war", "ear", "class",
                "db", "sqlite", "sqlite3",
            ];
            if binary_extensions.contains(&ext.as_ref()) {
                return true;
            }
        }

        // Check for minified JS/CSS files and other special cases
        let path_str = path.to_string_lossy();
        if path_str.ends_with(".min.js")
            || path_str.ends_with(".min.css")
            || path_str.ends_with(".bundle.js")
            || path_str.ends_with(".map")
        {
            return true;
        }

        false
    }

    pub fn grep_search(
        pattern: &str,
        include_pattern: Option<&str>,
        search_dir: Option<&Path>,
    ) -> Result<Vec<(PathBuf, usize, String)>> {
        let regex =
            Regex::new(pattern).with_context(|| format!("Invalid regex pattern: {pattern}"))?;

        let dir = search_dir.unwrap_or_else(|| Path::new("."));
        let include_regex = Self::create_file_pattern_filter(include_pattern);
        let mut matches = Vec::new();

        // Check if we should use repository ignore files
        let project_root = Self::find_project_root(dir);
        let use_repo_ignore = project_root
            .as_ref()
            .map(|root| Self::has_ignore_files(root))
            .unwrap_or(false);

        if use_repo_ignore {
            // Use the ignore crate's walker which respects .gitignore, etc.
            let walker = WalkBuilder::new(dir)
                .hidden(false)
                .standard_filters(true) // Respect .gitignore, .ignore, etc.
                .build();

            for entry in walker.flatten() {
                let path = entry.path();

                // Skip non-files
                if !entry.file_type().is_some_and(|ft| ft.is_file()) {
                    continue;
                }

                // Skip if doesn't match include pattern
                if let Some(ref include_regex) = include_regex {
                    if !include_regex.is_match(&path.to_string_lossy()) {
                        continue;
                    }
                }

                // Skip binary/generated files
                if Self::is_likely_binary_or_generated(path) {
                    continue;
                }

                // Try to search within the file
                if let Ok(file) = File::open(path) {
                    let reader = BufReader::new(file);
                    for (line_num, line_result) in reader.lines().enumerate() {
                        if let Ok(line) = line_result {
                            if regex.is_match(&line) {
                                matches.push((path.to_path_buf(), line_num + 1, line.clone()));
                            }
                        }
                    }
                }
            }
        } else {
            // Fall back to traditional walkdir with our hardcoded ignore patterns
            for entry in WalkDir::new(dir)
                .follow_links(true)
                .into_iter()
                .filter_entry(|e| !Self::should_skip_dir(e))
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
            {
                let path = entry.path();

                // Skip if doesn't match include pattern
                if let Some(ref include_regex) = include_regex {
                    if !include_regex.is_match(&path.to_string_lossy()) {
                        continue;
                    }
                }

                // Skip binary files and generated files
                if Self::is_likely_binary_or_generated(path) {
                    continue;
                }

                // Try to open file
                if let Ok(file) = File::open(path) {
                    let reader = BufReader::new(file);
                    for (line_num, line_result) in reader.lines().enumerate() {
                        if let Ok(line) = line_result {
                            if regex.is_match(&line) {
                                matches.push((path.to_path_buf(), line_num + 1, line.clone()));
                            }
                        }
                    }
                }
            }
        }

        // Sort by last modified time (most recent first)
        matches.sort_by(|a, b| {
            let a_modified = std::fs::metadata(&a.0).and_then(|m| m.modified()).ok();
            let b_modified = std::fs::metadata(&b.0).and_then(|m| m.modified()).ok();
            b_modified.cmp(&a_modified)
        });

        Ok(matches)
    }
}

fn glob_to_regex(glob_pattern: &str) -> String {
    let mut regex_pattern = String::new();

    for c in glob_pattern.chars() {
        match c {
            '*' => regex_pattern.push_str(".*"),
            '?' => regex_pattern.push('.'),
            '.' | '+' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '$' | '|' | '\\' => {
                regex_pattern.push('\\');
                regex_pattern.push(c);
            }
            _ => regex_pattern.push(c),
        }
    }

    format!("^{regex_pattern}$")
}
