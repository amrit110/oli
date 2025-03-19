# Oli TUI - AI Assistant Commands & Style Guide

## Build Commands
```bash
cargo build                   # Build the application
cargo run                     # Run the application
cargo clippy --all-targets    # Lint with clippy
cargo fmt                     # Format code 
cargo llvm-cov                # Run tests with coverage
cargo test                    # Run all tests
cargo test test_name_here     # Run a specific test
```

## Code Style Guidelines
- **Formatting**: Follow standard Rust formatting with `rustfmt`
- **Imports**: Group by crate (std first, then external, then internal)
- **Error Handling**: Use `anyhow::Result` with `?` operator pattern
- **Types**: Use strong typing with custom error types from `errors.rs`
- **Naming**: Snake case for functions/variables, Pascal case for types
- **UI Components**: Maintain consistent TUI design with existing patterns
- **Metal Acceleration**: Ensure compatibility when modifying model code
- **Documentation**: Add doc comments for public API functions and types
- **Testing**: Write tests for new functionality in `tests/` directory
