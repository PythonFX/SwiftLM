<!-- Generated: 2026-05-21 | Updated: 2026-05-21 -->

# SwiftLM

## Purpose
A native Swift inference server for Apple Silicon that serves MLX models via an OpenAI-compatible API. Built on Metal with no Python runtime — compiles to a single binary. Includes an iOS companion app (SwiftBuddy), speculative decoding (DFlash), and SSD expert streaming for massive MoE models.

## Key Files

| File | Description |
|------|-------------|
| `Package.swift` | SPM manifest — defines all targets, products, and dependencies |
| `build.sh` | Full build pipeline: submodules → cmake → Metal kernels → Swift release build |
| `build_swiftbuddy.sh` | SwiftBuddy macOS build helper |
| `run_benchmark.sh` | Interactive benchmark runner with context/memory profiling and regression tests |
| `Package.resolved` | Locked dependency versions |
| `.gitmodules` | Submodules: `mlx-swift` (SharpAI fork), `mlx-swift-lm` (SharpAI fork) |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `Sources/` | All Swift library and executable targets (see `Sources/AGENTS.md`) |
| `SwiftBuddy/` | iOS/macOS SwiftUI companion app (see `SwiftBuddy/AGENTS.md`) |
| `tests/` | Test suites, benchmarks, and integration test scripts (see `tests/AGENTS.md`) |
| `scripts/` | Build helpers, profiling, debugging utilities (see `scripts/AGENTS.md`) |
| `docs/` | Architecture docs, roadmaps, profiling results (see `docs/AGENTS.md`) |
| `.agents/` | Automated harness configs and workflow definitions for testing |
| `.github/workflows/` | CI: build, release, benchmark, dependency updates |
| `mlx-swift/` | Git submodule — SharpAI fork of Apple MLX Swift with SSD streaming patches |
| `mlx-swift-lm/` | Git submodule — SharpAI fork of Apple MLX LLM library with MoE/MTP support |

## For AI Agents

### Working In This Directory
- **Build**: `./build.sh` — handles submodules, cmake, Metal kernels, and Swift compilation. Requires `cmake` (Homebrew) and `MetalToolchain` (`xcodebuild -downloadComponent MetalToolchain`).
- **Run server**: `.build/release/SwiftLM --model <hf-model-id> --port 5413`
- **Run tests**: `swift test` (unit tests only). Integration tests are shell scripts in `tests/`.
- **Swift version**: swift-tools-version 5.9. Platforms: macOS 14+, iOS 17+.
- **Submodules are critical**: `mlx-swift` and `mlx-swift-lm` are SharpAI forks with custom C++ Metal kernels for SSD streaming and TurboQuant that don't exist in upstream Apple repos.

### Testing Requirements
- Unit tests: `swift test` (runs `SwiftLMTests` and `SwiftBuddyTests`)
- Integration tests: shell scripts in `tests/` (e.g., `test-server.sh`, `test-vision.sh`, `test-speculative.sh`)
- Benchmarks: `./run_benchmark.sh` or `python3 tests/run_4models_benchmark.py`

### Common Patterns
- Targets are organized as: shared libraries (`MLXInferenceCore`, `DFlash`) + executables (`SwiftLM`, `SwiftBuddy`)
- Model architecture support lives in the forked `mlx-swift-lm` submodule, not in this repo
- The `Server.swift` file (146KB) is the monolithic HTTP server handling all OpenAI API routes
- Metal kernel changes require rebuilding `mlx.metallib` via `build.sh` step 3

## Dependencies

### Internal (Submodule Forks)
- `mlx-swift` (SharpAI fork) — custom C++ primitives for SSD streaming, TurboQuant Metal kernels
- `mlx-swift-lm` (SharpAI fork) — GPU/CPU layer partitioning, MoE flush gate, MTP heads

### External
- `swift-transformers` (HuggingFace) — tokenizers + model download
- `hummingbird` — lightweight Swift HTTP server (OpenAI API layer)
- `swift-argument-parser` (Apple) — CLI flags
- `SwiftSoup` — HTML parsing (SwiftBuddy web tools)

<!-- MANUAL: Custom project notes can be added below -->
