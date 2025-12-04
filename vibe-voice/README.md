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

## Testing

```sh
curl -H "Content-Type: application/json" \
     -H "Authorization: Bearer <OICM-API-KEY>" \
     -d '{"text": "Speaker 1: this is a demo of vibevoice"}' \
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
