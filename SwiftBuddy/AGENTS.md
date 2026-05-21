<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-21 | Updated: 2026-05-21 -->

# SwiftBuddy

## Purpose
Native iOS/macOS SwiftUI companion app for SwiftLM. Downloads MLX models from HuggingFace and runs on-device inference via Metal. Features chat UI, model management, memory palace tools, and persona system.

## Key Files

| File | Description |
|------|-------------|
| `generate_xcodeproj.py` | Generates `.xcodeproj` with personal Team ID — run after cloning (git-ignored) |
| `SwiftBuddy/` | App source code (see `SwiftBuddy/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- **Build**: `python3 generate_xcodeproj.py` then open in Xcode. Requires Apple Developer Team for signing.
- The `.xcodeproj` is git-ignored because it contains personal Team IDs.
- Alternatively, build via SPM: `swift build --target SwiftBuddy` (macOS only, no iOS target).

### Testing Requirements
- `tests/SwiftBuddyTests/` — XCTest suite for SwiftBuddy
- Manual testing required for UI flows (Xcode + physical device/simulator)

### Common Patterns
- Uses `MLXInferenceCore` for model management (shared with CLI server)
- On-device inference runs directly via MLX Swift on iPhone/iPad Metal GPU
- Server mode: can also launch the SwiftLM CLI server as a subprocess

<!-- MANUAL: Custom project notes can be added below -->
