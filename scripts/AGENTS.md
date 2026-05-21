<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-21 | Updated: 2026-05-21 -->

# Scripts

## Purpose
Build helpers, profiling tools, debugging utilities, and small test scripts for development and CI.

## Key Files

| File | Description |
|------|-------------|
| `build.sh` | Master build script (also at repo root — this is the same or a variant) |
| `build_dmg.sh` | Creates a macOS DMG installer for SwiftLM releases |
| `build_swiftbuddy.sh` | Builds SwiftBuddy for macOS |
| `hf_discovery.py` | Discovers available MLX models on HuggingFace Hub |
| `measure_speed.py` | Quick inference speed measurement utility |
| `test_inference.sh` | Basic inference smoke test |
| `test_*.swift` | Small standalone Swift test scripts (assign, children, leaf, mirror, progress) |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `debugging/` | Debug utilities — token inspection, model loading recovery, weight checks |
| `profiling/` | Benchmark and profiling scripts — context/memory profiling, speculative decoding benchmarks |

## For AI Agents

### Working In This Directory
- `build.sh` is the canonical build entry point (symlinked from repo root)
- Python scripts require the model server to be running (they call the OpenAI API)
- Swift test scripts are standalone files — compile with `swiftc` or run directly

### Testing Requirements
- Profiling scripts produce output in `docs/profiling/`
- Debug scripts are ad-hoc — run individually as needed

<!-- MANUAL: Custom project notes can be added below -->
