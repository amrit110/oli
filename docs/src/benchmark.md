# oli tool benchmarks

This page contains the latest benchmark results for oli tool use performance.
These benchmarks are automatically updated with each new PR.

## Tool Performance Overview

The benchmark test measures how efficiently oli's tools operate when used with local
Ollama models. The benchmark evaluates every tool's performance using simple test cases.

## Latest Benchmark Results

_This section is automatically updated by CI/CD pipelines._

<!-- BENCHMARK_RESULTS -->
## Latest Results (as of 2025-05-22 19:58:14 UTC)

| Category | Details |
|----------|---------|
| Model | `qwen2.5-coder:7b` |
| Tool Benchmark Time | 198662 ms |
| Tool Tests | 4/8 tests passed |

### Tool Performance Tests
- [x] test_read_file_tool_with_llm (66515ms (66.51s))
- [ ] test_glob_tool_with_llm (15175ms (15.17s))
- [x] test_grep_tool_with_llm (12120ms (12.12s))
- [ ] test_ls_tool_with_llm (22861ms (22.86s))
- [x] test_document_symbol_tool_with_llm (430ms (.43s))
- [ ] test_edit_tool_with_llm (13052ms (13.05s))
- [x] test_bash_tool_with_llm (35670ms (35.67s))
- [ ] test_write_tool_with_llm (32650ms (32.65s))

<!-- END_BENCHMARK_RESULTS -->
