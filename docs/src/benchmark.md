# oli tool benchmarks

This page contains the latest benchmark results for oli tool use performance.
These benchmarks are automatically updated with each new PR.

## Tool Performance Overview

The benchmark test measures how efficiently oli's tools operate when used with local
Ollama models. The benchmark evaluates every tool's performance using simple test cases.

## Latest Benchmark Results

_This section is automatically updated by CI/CD pipelines._

<!-- BENCHMARK_RESULTS -->
## Latest Results (as of 2025-05-13 03:08:08 UTC)

| Category | Details |
|----------|---------|
| Model | `qwen2.5-coder:7b` |
| Tool Benchmark Time | 152995 ms |
| Tool Tests | 4/8 tests passed |

### Tool Performance Tests
- [x] test_read_file_tool_with_llm (49305ms (49.30s))
- [x] test_glob_tool_with_llm (13715ms (13.71s))
- [ ] test_grep_tool_with_llm (19314ms (19.31s))
- [ ] test_ls_tool_with_llm (18222ms (18.22s))
- [x] test_document_symbol_tool_with_llm (406ms (.40s))
- [ ] test_edit_tool_with_llm (20331ms (20.33s))
- [x] test_bash_tool_with_llm (9812ms (9.81s))
- [ ] test_write_tool_with_llm (21843ms (21.84s))

<!-- END_BENCHMARK_RESULTS -->
