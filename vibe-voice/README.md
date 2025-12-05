## VibeVoice Docker Deployment

## Prerequisites

- Application must run on port `8080`
- Health Check endpoint must be `/health-check`

## Setup

- [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)

- Download VibeVoice model to `models` directory
  ```sh
  uvx hf download microsoft/VibeVoice-1.5B --local-dir ./models/VibeVoice-1.5B
  ```

- Download `Qwen/Qwen2.5-1.5B` model to `models` directory 

  ```sh 
    uvx hf download Qwen/Qwen2.5-1.5B --local-dir ./models/Qwen2.5-1.5B
  ```
- Run the script to update `preprocessor_config.json` to set path to `Qwen/Qwen2.5-1.5B` model: 

  ```sh
  python3 scripts/run.py
  ```

- Build the image for CPU/Nvidia GPU: 

  ```sh 
  # Example
  OPERATOR=cpu docker build -t <IMAGE-TAG> .

  # Build for CPU
  OPERATOR=cpu docker build -t oicm-vibe-voice:cpu .

  # Build for Nvidia GPU
  OPERATOR=nvidia docker build -t oicm-vibe-voice:nvidia .
  ```

- Push to Container Registry

  ```sh
  # Login to Container Registry 
  docker login -u <username>

  # Push the image
  docker push <IMAGE-TAG>

  # CPU
  docker push oicm-vibe-voice:cpu

  # Nvidia
  docker push oicm-vibe-voice:nvidia
  ```

## API Payload Format

### Request

**Endpoint:** `POST /tts`

**Payload:**
```json
{
  "text": "Speaker 1: this is a demo of vibevoice. Speaker 2: checking the demo of vibe voice",
  "voice_samples": ["<base64-encoded-wav-1>", "<base64-encoded-wav-2>"]
}
```

**Fields:**
- `text` (required): Text to synthesize into speech
- `voice_samples` (optional): Array of voice samples in one of these formats:
  - Omit field to use default voices
  - Array of base64-encoded WAV strings: `["base64string1", "base64string2"]`
  - Array of objects with audio_base64 key: `[{"audio_base64": "base64string"}]`
  - Each entry resonates to voice that will be cloned, and based on the index of the voice, use `Speaker <idx>` to use the voice.

**Note:** Voice samples are automatically resampled to 24kHz and converted to mono if needed.

### Response

```json
{
  "audio_content": "<base64-encoded-wav>",
  "content_type": "audio/wav"
}
```

## Testing

**Using default voices:**
```sh
curl -H "Content-Type: application/json" \
     -H "Authorization: Bearer <OICM-API-KEY>" \
     -d '{"text": "Speaker 1: this is a demo of vibevoice"}' \
     https://<OICM-INFERENCE-PROXY-URL>/tts | \
     jq -r '.audio_content' | \
     base64 -d > output.wav
```

**Using custom voice sample:**
```sh
VOICE_BASE64=$(base64 -i voice.wav)
curl -H "Content-Type: application/json" \
     -H "Authorization: Bearer <OICM-API-KEY>" \
     -d '{"text": "Hello world", "voice_samples": ["'$VOICE_BASE64'"]}' \
     https://<OICM-INFERENCE-PROXY-URL>/tts | \
     jq -r '.audio_content' | \
     base64 -d > output.wav
```

## Development

Setup virtual environment

```sh 
uv venv --python 3.12

# Activate venv
source .venv/bin/active

# Install dependencies
uv sync --frozen --no-install-project

uv pip install -e ./vibevoice
```

Run the server 

```sh 
uv run server.py
```
