<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-21 | Updated: 2026-05-21 -->

# Profiling Scripts

## Purpose
Benchmark and profiling tools for measuring inference speed, memory usage, and speculative decoding performance across context lengths.

## Key Files

| File | Description |
|------|-------------|
| `profile_runner.py` | Main profiling runner — measures TPS and RAM across context lengths (512, 40K, 100K). Results saved to `docs/profiling/` |
| `profile_speculative.py` | Profiles speculative decoding performance — acceptance rates, TPS improvement |
| `fp8_mtp_harness.py` | MTP (Multi-Token Prediction) benchmark harness |
| `bench_35b.sh` | Benchmark script for 35B parameter models |
| `bench_coder_next.sh` | Benchmark for code generation models |

## For AI Agents

### Working In This Directory
- `profile_runner.py` is the primary tool: `python3 -u scripts/profiling/profile_runner.py --model <model> --contexts "512,40000,100000"`
- Requires the SwiftLM server to be running (or launches it automatically)
- Output goes to `docs/profiling/profiling_results_<hostname>.md`

<!-- MANUAL: Custom project notes can be added below -->
