# oli tool benchmarks

This page contains the latest benchmark results for oli tool use performance.
These benchmarks are automatically updated with each new PR.

## Tool Performance Overview

The benchmark test measures how efficiently oli's tools operate when used with local
Ollama models. The benchmark evaluates every tool's performance using simple test cases.

## Latest Benchmark Results

_This section is automatically updated by CI/CD pipelines._

<!-- BENCHMARK_RESULTS -->
## Latest Results (as of 2025-07-23 16:52:16 UTC)

| Category | Details |
|----------|---------|
| Model | `qwen2.5-coder:7b` |
| Tool Benchmark Time | 205713 ms |
| Tool Tests | 3/8 tests passed |

### Tool Performance Tests
- [x] test_read_file_tool_with_llm (71288ms (71.28s))
- [x] test_glob_tool_with_llm (15233ms (15.23s))
- [ ] test_grep_tool_with_llm (24157ms (24.15s))
- [ ] test_ls_tool_with_llm (21020ms (21.02s))
- [x] test_document_symbol_tool_with_llm (2830ms (2.83s))
- [ ] test_edit_tool_with_llm (13806ms (13.80s))
- [ ] test_bash_tool_with_llm (33751ms (33.75s))
- [ ] test_write_tool_with_llm (23580ms (23.58s))

<!-- END_BENCHMARK_RESULTS -->
