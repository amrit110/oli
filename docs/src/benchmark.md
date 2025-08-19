# oli tool benchmarks

This page contains the latest benchmark results for oli tool use performance.
These benchmarks are automatically updated with each new PR.

## Tool Performance Overview

The benchmark test measures how efficiently oli's tools operate when used with local
Ollama models. The benchmark evaluates every tool's performance using simple test cases.

## Latest Benchmark Results

_This section is automatically updated by CI/CD pipelines._

<!-- BENCHMARK_RESULTS -->
## Latest Results (as of 2025-08-19 14:39:53 UTC)

| Category | Details |
|----------|---------|
| Model | `qwen2.5-coder:7b` |
| Tool Benchmark Time | 54071 ms |
| Tool Tests | 1/8 tests passed |

### Tool Performance Tests
- [ ] test_read_file_tool_with_llm (50856ms (50.85s))
- [ ] test_glob_tool_with_llm (493ms (.49s))
- [ ] test_grep_tool_with_llm (455ms (.45s))
- [ ] test_ls_tool_with_llm (433ms (.43s))
- [x] test_document_symbol_tool_with_llm (446ms (.44s))
- [ ] test_edit_tool_with_llm (450ms (.45s))
- [ ] test_bash_tool_with_llm (443ms (.44s))
- [ ] test_write_tool_with_llm (433ms (.43s))

<!-- END_BENCHMARK_RESULTS -->
