<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-21 | Updated: 2026-05-21 -->

# MLXInferenceCore

## Purpose
Shared inference library used by both the SwiftLM CLI server (macOS) and the SwiftBuddy iOS app. Handles model download, storage, catalog management, architecture probing, audio processing, and the core inference engine.

## Key Files

| File | Description |
|------|-------------|
| `InferenceEngine.swift` | Core inference orchestration — token generation, KV cache management, prefill pipeline |
| `ModelStorage.swift` | On-device model storage — filesystem layout, deletion, space management, migration |
| `ModelDownloader.swift` | HuggingFace model download with progress tracking and resume |
| `ModelDownloadManager.swift` | Download queue manager — concurrent downloads, cancellation, priority |
| `ModelCatalog.swift` | Curated model catalog with RAM fit indicators for device capabilities |
| `HFModelSearch.swift` | HuggingFace Hub API search — find any `mlx-community` model by name |
| `ModelArchitectureProbe.swift` | Probe model config to determine architecture type (dense, MoE, VLM, ALM) |
| `GenerationConfig.swift` | Per-request generation parameters — temperature, top-p, top-k, repetition penalty |
| `ChatMessage.swift` | OpenAI-compatible chat message types |
| `OpenAIPayloads.swift` | Request/response types matching the OpenAI API spec |
| `CLICommandBuilder.swift` | Build CLI command strings from model configs (for SwiftBuddy server launch) |
| `SSDStreamingRecovery.swift` | Recovery logic for SSD streaming failures — retry, fallback, error reporting |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `ALM/` | Audio-Language Model processing — STFT, Mel spectrogram, multimodal fusion (see `ALM/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- This target has **StrictConcurrency** enabled — all code must be concurrency-safe.
- Changes here affect **both** macOS CLI and iOS app. Test on both platforms.
- `InferenceEngine.swift` is the central piece — it wraps the MLX evaluation loop.
- `ModelStorage.swift` defines the on-disk model directory structure (`~/Library/Application Support/SwiftLM/`).

### Testing Requirements
- `SwiftLMTests` and `SwiftBuddyTests` both import this target
- Build with `swift build` — concurrency warnings are errors

### Common Patterns
- All model paths can be either HuggingFace IDs (auto-download) or local filesystem paths
- The inference engine abstracts over text, vision, and audio modalities

## Dependencies

### Internal
- `mlx-swift` (MLX), `mlx-swift-lm` (MLXLLM, MLXVLM, MLXLMCommon, MLXHuggingFace)

### External
- `Hub`, `Tokenizers` (from `swift-transformers`) — HuggingFace model registry and tokenization

<!-- MANUAL: Custom project notes can be added below -->
