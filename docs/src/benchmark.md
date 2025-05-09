# oli tool benchmarks

This page contains the latest benchmark results for oli tool use performance.
These benchmarks are automatically updated with each new PR.

## Tool Performance Overview

The benchmark test measures how efficiently oli's tools operate when used with local
Ollama models. The benchmark evaluates every tool's performance using simple test cases.

## Latest Benchmark Results

_This section is automatically updated by CI/CD pipelines._

<!-- BENCHMARK_RESULTS -->
## Latest Results (as of 2025-05-02 13:27:40 UTC)

| Category | Details |
|----------|---------|
| Model | `qwen2.5-coder:7b` |
| Tool Benchmark Time | 141433 ms |
| Tool Tests | 7/7 tests passed |

### Tool Performance Tests
- [x] test_read_file_tool_with_llm (54216ms (54.21s))
- [x] test_glob_tool_with_llm (11454ms (11.45s))
- [x] test_grep_tool_with_llm (12549ms (12.54s))
- [x] test_ls_tool_with_llm (11595ms (11.59s))
- [x] test_edit_tool_with_llm (12854ms (12.85s))
- [x] test_bash_tool_with_llm (29415ms (29.41s))
- [x] test_write_tool_with_llm (9305ms (9.30s))

<!-- END_BENCHMARK_RESULTS -->
