<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-21 | Updated: 2026-05-21 -->

# Sources

## Purpose
Contains all Swift targets: the shared inference library, the CLI HTTP server, speculative decoding engine, and utility executables. Each subdirectory maps 1:1 to a SwiftPM target defined in `Package.swift`.

## Subdirectories

| Directory | Target Type | Purpose |
|-----------|-------------|---------|
| `SwiftLM/` | Executable | CLI HTTP server — OpenAI-compatible API, model loading, routing (see `SwiftLM/AGENTS.md`) |
| `MLXInferenceCore/` | Library | Shared inference engine for SwiftLM and SwiftBuddy — model management, storage, download, ALM (see `MLXInferenceCore/AGENTS.md`) |
| `DFlash/` | Library | Block-diffusion speculative decoding — DFlash kernels, draft models, runtime engine (see `DFlash/AGENTS.md`) |
| `DFlashKernelBench/` | Executable | Micro-benchmark tool for DFlash Metal kernel performance |
| `SwiftLMTestSTFT/` | Executable | STFT audio profiling test script for audio-language models |
| `Gemma4MTPBench/` | Executable | MTP (Multi-Token Prediction) benchmark target |

## For AI Agents

### Working In This Directory
- `MLXInferenceCore` is shared between macOS CLI (`SwiftLM`) and iOS app (`SwiftBuddy`). Changes here affect both.
- `DFlash` depends on MLX primitives but is decoupled from the server — it provides draft models for speculative decoding.
- `SwiftLM/Server.swift` is the monolithic entry point (146KB). All HTTP routes, model lifecycle, and inference orchestration live there.
- `DFlashKernelBench` and `SwiftLMTestSTFT` are standalone executables for targeted profiling — not part of the main server.

### Testing Requirements
- `swift build` compiles all targets
- `swift test` runs `SwiftLMTests` and `SwiftBuddyTests` (both depend on targets here)

### Common Patterns
- Strict concurrency is enabled for `MLXInferenceCore` (`StrictConcurrency` experimental feature)
- `DFlashKernelsOptimized.swift` is excluded from the build (listed in `exclude`) — it's an in-progress optimization
- Model-specific DFlash adapters use Swift extensions on existing model types (e.g., `Llama+DFlash.swift`, `Qwen3+DFlash.swift`)

## Dependencies

### Internal
- All targets depend on the `mlx-swift` submodule (MLX framework)
- `SwiftLM` depends on `MLXInferenceCore` and `DFlash`
- `DFlash` depends on `MLXLLM` and `MLXLMCommon` from the `mlx-swift-lm` submodule

<!-- MANUAL: Custom project notes can be added below -->
