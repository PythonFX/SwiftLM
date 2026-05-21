<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-21 | Updated: 2026-05-21 -->

# Tests

## Purpose
Test suites, integration test scripts, and benchmark tools. Mix of XCTest targets (compiled via SPM) and shell/Python integration tests that exercise the running server.

## Key Files

| File | Description |
|------|-------------|
| `test-server.sh` | Comprehensive server integration test (47KB) — exercises all API endpoints, streaming, error handling |
| `test-vision.sh` | VLM integration test — image inputs, base64 encoding, vision model responses |
| `test-speculative.sh` | Speculative decoding integration test — dual-model loading, acceptance rate validation |
| `test-speculative-eval.sh` | Speculative decoding evaluation — correctness and quality checks |
| `test-dflash.sh` | DFlash block-diffusion integration test |
| `test-audio.sh` | Audio-language model integration test |
| `test-omni.sh` | Omni (audio+text) model integration test |
| `test-opencode.sh` | Code generation model test |
| `test-graph.sh` | Knowledge graph / memory palace test |
| `run_benchmarks.py` | Automated benchmark runner |
| `run_4models_benchmark.py` | Benchmark 4 specific models — TPS and GPU memory |
| `run_4models_100k.py` | 100K context benchmark for 4 models |
| `run_matrix.py` | Configuration matrix benchmark — Vanilla/SSD/TurboQuant/SSD+TQ across contexts |
| `run_passkey_100k.py` | Passkey retrieval test at 100K context (measures long-context retention) |
| `run_extreme_context.sh` | Extreme context length stress test |
| `system_monitor.py` / `system_monitor.sh` | Real-time system memory/CPU monitoring during tests |
| `test_turbo_quant.cpp` | C++ correctness test for TurboQuant quantization tables |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `SwiftLMTests/` | XCTest suite for SwiftLM target |
| `SwiftBuddyTests/` | XCTest suite for SwiftBuddy target |
| `DFlash/` | DFlash-specific test fixtures and intermediate dumps |
| `fixtures/` | Shared test fixtures (JSON payloads, model configs) |
| `fixtures/omni/` | Omni model test fixtures (audio samples, configs) |
| `sandbox/` | Experimental and work-in-progress test scripts |

## For AI Agents

### Working In This Directory
- **XCTest**: `swift test` runs `SwiftLMTests` and `SwiftBuddyTests`
- **Integration tests**: Shell scripts require the server to be running. Most scripts can launch the server automatically.
- **Benchmarks**: Python scripts call the server's OpenAI API to measure performance
- Shell tests use `curl` against the server's HTTP API
- `test-server.sh` is the most comprehensive integration test — run it first for smoke testing

### Testing Requirements
- Integration tests need a downloaded model (auto-downloaded if not cached)
- Tests may take significant time (100K context benchmarks)
- `system_monitor.sh` should run in a separate terminal during memory-sensitive tests

### Common Patterns
- Benchmarks output markdown tables to `docs/profiling/`
- Integration tests check for specific HTTP status codes and response shapes
- `run_extreme_context.sh` and `run_passkey_100k.py` validate memory stability at extreme context lengths

<!-- MANUAL: Custom project notes can be added below -->
