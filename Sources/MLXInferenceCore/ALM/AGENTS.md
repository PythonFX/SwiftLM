<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-21 | Updated: 2026-05-21 -->

# ALM (Audio-Language Models)

## Purpose
Audio processing pipeline for audio-language models (e.g., Gemma 4 Omni). Handles audio ingestion, STFT spectrogram computation, and multimodal fusion with text inputs.

## Key Files

| File | Description |
|------|-------------|
| `ALMTypeRegistry.swift` | Registry of supported audio model types and their configurations |
| `AudioTTS.swift` | Text-to-speech audio output handling and AVFoundation integration |
| `MultimodalFusionProcessor.swift` | Fuses audio embeddings with text embeddings for multimodal model input |
| `Whisper.swift` | Whisper-based audio encoding — WAV extraction via AVFoundation, vDSP STFT, Mel spectrogram (16kHz, 128 bins) |

## For AI Agents

### Working In This Directory
- Audio pipeline: WAV extraction → vDSP STFT → Mel spectrogram → embedding fusion
- Uses Apple's `AVFoundation` and `Accelerate` (vDSP) frameworks for audio processing
- Currently only supports `gemma-4-e4b` variants (audio tower is model-specific)

### Testing Requirements
- Integration test: `tests/test-audio.sh` and `tests/test-omni.sh`
- The `SwiftLMTestSTFT` executable in `Sources/SwiftLMTestSTFT/` provides standalone STFT profiling

### Common Patterns
- Audio data arrives as OpenAI-spec `input_audio` payloads (base64-encoded)
- STFT parameters are hardcoded to model requirements (16kHz sample rate, 128 Mel bins)

<!-- MANUAL: Custom project notes can be added below -->
