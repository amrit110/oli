# oli tool benchmarks

This page contains the latest benchmark results for oli tool use performance.
These benchmarks are automatically updated with each new PR.

## Tool Performance Overview

The benchmark test measures how efficiently oli's tools operate when used with local
Ollama models. The benchmark evaluates every tool's performance using simple test cases.

## Latest Benchmark Results

_This section is automatically updated by CI/CD pipelines._

<!-- BENCHMARK_RESULTS -->
## Latest Results (as of 2025-08-27 13:34:30 UTC)

| Category | Details |
|----------|---------|
| Model | `qwen2.5-coder:7b` |
| Tool Benchmark Time | 3531 ms |
| Tool Tests | 1/8 tests passed |

### Tool Performance Tests
- [ ] test_read_file_tool_with_llm (440ms (.44s))
- [ ] test_glob_tool_with_llm (456ms (.45s))
- [ ] test_grep_tool_with_llm (437ms (.43s))
- [ ] test_ls_tool_with_llm (430ms (.43s))
- [x] test_document_symbol_tool_with_llm (426ms (.42s))
- [ ] test_edit_tool_with_llm (431ms (.43s))
- [ ] test_bash_tool_with_llm (432ms (.43s))
- [ ] test_write_tool_with_llm (431ms (.43s))

<!-- END_BENCHMARK_RESULTS -->
