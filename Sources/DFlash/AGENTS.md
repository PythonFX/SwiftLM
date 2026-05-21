<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-21 | Updated: 2026-05-21 -->

# DFlash — Speculative Decoding

## Purpose
Block-diffusion speculative decoding engine. Loads a small DFlash draft model alongside the main model to generate candidate token blocks and verify them in bulk, accelerating in-RAM inference. Includes custom Metal kernels for draft generation and a recurrent-state rollback cache for hybrid architectures.

## Key Files

| File | Description |
|------|-------------|
| `DFlashEngine.swift` | Top-level engine — coordinates draft generation, verification, and acceptance |
| `DFlashRuntime.swift` | Runtime execution — draft block generation, token acceptance, rejection sampling |
| `DFlashKernels.swift` | Custom Metal/GPU kernels for DFlash operations (quantization, projection) |
| `DFlashKernelsOptimized.swift` | Optimized kernel variants (excluded from build — in-progress) |
| `DFlashDraftModel.swift` | Draft model loading and configuration — architecture-specific weight handling |
| `DFlashDraftBackend.swift` | Backend abstraction for different draft model architectures |
| `DFlashDraftRegistry.swift` | Registry of supported DFlash draft model types |
| `DFlashKernelProvider.swift` | Kernel selection and configuration provider |
| `DFlashIntermediateDumper.swift` | Debug utility — dumps intermediate tensors for correctness verification |
| `RecurrentRollbackCache.swift` | Checkpoint/restore for Mamba recurrent state — enables speculative decoding on hybrid Attention+Mamba architectures |

## For AI Agents

### Working In This Directory
- `DFlashKernelsOptimized.swift` is **excluded** from the build (`Package.swift` exclude list) — do not reference it.
- Draft model adapters are in `Server.swift` as extensions (e.g., `DeepseekV3DFlash.swift`, `KimiLinearDFlash.swift`, `Llama+DFlash.swift`, `Qwen3+DFlash.swift`).
- `RecurrentRollbackCache` is critical for Mamba-hybrid models (Qwen3.5) — Mamba state cannot be partially trimmed like attention KV caches.

### Testing Requirements
- Integration tests: `tests/test-dflash.sh`, `tests/test-speculative.sh`, `tests/test-speculative-eval.sh`
- `tests/DFlash/` contains DFlash-specific test fixtures and intermediates

### Common Patterns
- Draft models are loaded separately from the main model via `--dflash` flag
- Block size is configurable via `--dflash-block-size`
- Acceptance uses greedy (`argMax`) decoding — stochastic sampling is tracked as a known limitation (see README)

## Dependencies

### Internal
- `MLXLLM`, `MLXLMCommon` (from `mlx-swift-lm` submodule)
- `MLX` (from `mlx-swift` submodule)

<!-- MANUAL: Custom project notes can be added below -->
