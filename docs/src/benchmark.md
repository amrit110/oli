# oli tool benchmarks

This page contains the latest benchmark results for oli tool use performance.
These benchmarks are automatically updated with each new PR.

## Tool Performance Overview

The benchmark test measures how efficiently oli's tools operate when used with local
Ollama models. The benchmark evaluates every tool's performance using simple test cases.

## Latest Benchmark Results

_This section is automatically updated by CI/CD pipelines._

<!-- BENCHMARK_RESULTS -->
## Latest Results (as of 2025-05-09 15:48:12 UTC)

| Category | Details |
|----------|---------|
| Model | `qwen2.5-coder:7b` |
| Tool Benchmark Time | 184772 ms |
| Tool Tests | 3/8 tests passed |

### Tool Performance Tests
- [ ] test_read_file_tool_with_llm (46225ms (46.22s))
- [ ] test_glob_tool_with_llm (24098ms (24.09s))
- [x] test_grep_tool_with_llm (21435ms (21.43s))
- [x] test_ls_tool_with_llm (19882ms (19.88s))
- [x] test_document_symbol_tool_with_llm (402ms (.40s))
- [ ] test_edit_tool_with_llm (12236ms (12.23s))
- [ ] test_bash_tool_with_llm (35602ms (35.60s))
- [ ] test_write_tool_with_llm (24844ms (24.84s))

<!-- END_BENCHMARK_RESULTS -->
