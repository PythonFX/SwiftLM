<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-21 | Updated: 2026-05-21 -->

# SwiftLM CLI Server

## Purpose
The main executable — an OpenAI-compatible HTTP inference server powered by Metal. Handles model loading, inference orchestration, KV cache management, speculative decoding, and all API routing.

## Key Files

| File | Description |
|------|-------------|
| `Server.swift` | Monolithic server (146KB) — HTTP routes, model lifecycle, inference pipeline, streaming, all API logic |
| `ModelProfiler.swift` | Runtime profiling of model evaluation — per-layer timing, memory tracking, TPS calculation |
| `Calibrator.swift` | Wisdom auto-calibration — determines optimal `--gpu-layers` based on available hardware memory |

## For AI Agents

### Working In This Directory
- `Server.swift` is intentionally monolithic. It handles: argument parsing, model loading, all `/v1/chat/completions` variants (text, vision, audio), streaming SSE, KV cache lifecycle, speculative decoding coordination, TurboQuant toggling, and SSD streaming configuration.
- When modifying API behavior, search `Server.swift` for the relevant route handler. The file is large but logically organized top-to-bottom.
- `ModelProfiler` and `Calibrator` are utilities invoked during startup or per-request — not HTTP handlers.

### Testing Requirements
- Integration tests in `tests/` exercise the server end-to-end via HTTP (`test-server.sh`, `test-vision.sh`, etc.)
- No unit tests exist for this target directly — test via `SwiftLMTests` or integration scripts

### Common Patterns
- CLI flags are defined inline in `Server.swift` using `ArgumentParser`
- Model loading delegates to `MLXInferenceCore` for shared logic
- Streaming uses Hummingbird's async response bodies with SSE formatting

## Dependencies

### Internal
- `MLXInferenceCore` — model download, storage, architecture probing, inference engine
- `DFlash` — speculative decoding engine
- `mlx-swift` (MLX), `mlx-swift-lm` (MLXLLM, MLXVLM, MLXLMCommon, MLXHuggingFace)

### External
- `Hummingbird` — HTTP server framework
- `ArgumentParser` — CLI argument parsing
- `Transformers` (HuggingFace) — tokenizers

<!-- MANUAL: Custom project notes can be added below -->
