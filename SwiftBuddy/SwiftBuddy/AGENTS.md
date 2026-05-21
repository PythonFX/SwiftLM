<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-21 | Updated: 2026-05-21 -->

# SwiftBuddy App Source

## Purpose
The SwiftUI app source — views, view models, personas, theming, and the app entry point.

## Key Files

| File | Description |
|------|-------------|
| `SwiftBuddyApp.swift` | App entry point — SwiftUI lifecycle, tab navigation, window management |
| `Theme.swift` | Design system — colors, typography, component styling for the app |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `ViewModels/` | MVVM view models — chat, server management, model download, memory palace |
| `Views/` | SwiftUI views — chat, models, settings, memory palace, inspector |
| `Personas/` | JSON persona definitions (e.g., `Lumina.json` — excluded from build) |
| `Assets.xcassets/` | App icons and accent colors |

## For AI Agents

### Working In This Directory
- Architecture: MVVM with SwiftUI. ViewModels handle all business logic; Views are declarative.
- `ServerManager.swift` manages the SwiftLM CLI server as a subprocess on macOS.
- `ChatViewModel.swift` is the primary view model — handles message flow, streaming responses, model interaction.
- `MemoryPalaceService.swift` + `GraphPalaceService.swift` implement knowledge graph / memory palace features.
- `ExtractionService.swift` extracts structured data from model responses using SwiftSoup.

### Testing Requirements
- `tests/SwiftBuddyTests/` contains XCTest cases
- UI changes require manual verification in Xcode simulator or on-device

### Common Patterns
- Personas are JSON files loaded at runtime via `PersonaLoader.swift`
- Web tools (browsing, search) use `WebToolService.swift` with SwiftSoup HTML parsing
- `SystemMonitorService.swift` tracks device memory/CPU for model fit assessment

## Dependencies

### Internal
- `MLXInferenceCore` — shared inference library

### External
- `Hummingbird` — embedded HTTP server (for server mode)
- `SwiftSoup` — HTML parsing for web tool extraction

<!-- MANUAL: Custom project notes can be added below -->
