## Custom models docker images for OICM platform

This repo contains examples for the custom docker images deployment.

These images work with OICM 1.15.0 or lower. To use these images in OICM 1.16.0 and later, update the health check endpoint from `/health-check` to `/health`.

### Available Services

#### Speaker Diarization 3.1

Location: `speaker-diarization-3.1/`

Endpoints:

- `GET /health-check` - Health check endpoint
- `POST /v1/audio/diarization` - Speaker diarization endpoint (supports JSON URL, multipart file upload, or raw bytes)

See [speaker-diarization-3.1/README.md](speaker-diarization-3.1/README.md) for detailed documentation.

#### OpenAI Whisper

Location: `openai-whisper/`

Endpoints:

- `GET /health-check` - Health check endpoint
- `POST /v1/audio/transcriptions` - Speech-to-text transcription (OpenAI Whisper compatible, with segments)

See [openai-whisper/README.md](openai-whisper/README.md) for detailed documentation.

#### Omnilingual ASR

Location: `omniasr-server/`

FastAPI-based ASR server for [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) with an OpenAI Whisper-compatible API.

Endpoints:

- `GET /health-check` - Health check endpoint
- `GET /v1/models` - List available models
- `POST /v1/audio/transcriptions` - Speech-to-text transcription

See [omniasr-server/README.md](omniasr-server/README.md) for detailed documentation.

#### DeepSeek-OCR

Location: `deepseek-ocr/`

DeepSeek-OCR served via vLLM behind nginx, exposing an OpenAI-compatible API.

Endpoints:

- `GET /health-check` - Health check endpoint
- `POST /v1/*` - OpenAI-compatible vLLM endpoints (e.g. chat/completions)

#### Qwen Image Edit

Location: `qwen-image-edit-image/`

Endpoints:

- `GET /health-check` - Health check endpoint
- `POST /v1/images/edits` - Image editing endpoint

See [qwen-image-edit-image/README.md](qwen-image-edit-image/README.md) for detailed documentation.

#### VibeVoice (Text-to-Speech)

Location: `vibe-voice/`

Text-to-speech service based on [VibeVoice](https://github.com/microsoft/VibeVoice).

Endpoints:

- `GET /health-check` - Health check endpoint
- `POST /tts` - Text-to-speech synthesis (supports voice samples)

See [vibe-voice/README.md](vibe-voice/README.md) for detailed documentation.

#### Deepfake Detector

Location: `deepfake-detector/`

Audio deepfake classification service.

Endpoints:

- `GET /health-check` - Health check endpoint
- `POST /classify` - Classify an audio file as real or deepfake (accepts JSON with `audio_url`)

See [deepfake-detector/README.md](deepfake-detector/README.md) for detailed documentation.
