<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-21 | Updated: 2026-05-21 -->

# Debugging Scripts

## Purpose
Ad-hoc debugging utilities for inspecting model tokens, weights, loading behavior, and recovery scenarios.

## Key Files

| File | Description |
|------|-------------|
| `list_keys.py` | Lists safetensors keys in a model checkpoint |
| `model_loading_recovery_harness.sh` | Tests model loading failure recovery paths |
| `test_tokens.py` | Tokenizer debugging — encode/decode inspection |
| `test_tokens.swift` | Swift tokenizer test counterpart |
| `test_update.swift` | Tests model weight update/loading flows |
| `test_weights.swift` | Inspects model weight shapes and dtypes |

## For AI Agents

### Working In This Directory
- Scripts are standalone — run directly against a downloaded model path
- Python scripts may need `pip install` for HuggingFace dependencies

<!-- MANUAL: Custom project notes can be added below -->
