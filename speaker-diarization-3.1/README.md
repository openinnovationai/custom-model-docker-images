## Speaker Diarization 3.1 (FastAPI + pyannote)

Production-ready HTTP service that performs speaker diarization using `pyannote/speaker-diarization-3.1`.

The service exposes a simple REST API and can be built for CPU-only or NVIDIA GPUs. The container pre-downloads the Hugging Face model at build time to enable stable, offline-friendly deployments.

### Contents
- Endpoint reference
- Quick start (CPU and NVIDIA GPU)
- Build with `build.sh`
- Direct `docker build` commands
- Run and test examples
- Notes and troubleshooting

---

## Endpoint reference

Base URL: `http://localhost:8080`

- `GET /health-check`
  - Returns `200 {"status":"ok"}` when the model is loaded. Returns `503` if the model is not loaded.

- `POST /diarize`
  - Accepts one of the following input modes:
    1) JSON: `{ "url": "https://example.com/audio.wav" }`
    2) Multipart form: field name `file` (e.g. upload `file=@audio.wav`)
    3) Raw bytes: request body containing audio bytes with `Content-Type: application/octet-stream`
  - Response: JSON array of diarization segments (RTTM-like fields):
    - `type` (always `"SPEAKER"`)
    - `file_id`
    - `channel_id`
    - `turn_onset` (seconds, float)
    - `turn_duration` (seconds, float)
    - `orthography_field`
    - `speaker_type`
    - `speaker_name`
    - `confidence_score` (string number or `"<NA>"`)
    - `signal_lookahead_time`

Example minimal response item:
```json
{
  "type": "SPEAKER",
  "file_id": "audio",
  "channel_id": 0,
  "turn_onset": 1.23,
  "turn_duration": 2.34,
  "orthography_field": "SPEAKER_00",
  "speaker_type": "speaker",
  "speaker_name": "SPEAKER_00",
  "confidence_score": "0.987654",
  "signal_lookahead_time": "0"
}
```

---

## Requirements

- Docker
- For GPU builds/runs: NVIDIA GPU + recent NVIDIA driver + `nvidia-container-toolkit`
- A Hugging Face access token with permission to download `pyannote/speaker-diarization-3.1` (required at build time to pre-download the model)
  - Export as env var: `export HF_TOKEN=hf_xxx`

The image exposes port `8080`.

---

## Quick start

### CPU build and run
```bash
cd speaker-diarization-3.1
export HF_TOKEN=hf_your_token
./build.sh  # defaults to OPERATOR=cpu; builds image tag: pyannote-cpu

docker run --rm -p 8080:8080 pyannote-cpu
```

### NVIDIA GPU build and run
```bash
cd speaker-diarization-3.1
export HF_TOKEN=hf_your_token
OPERATOR=nvidia ./build.sh  # builds image tag: pyannote-nvidia

docker run --rm --gpus all -p 8080:8080 pyannote-nvidia
```

At runtime the service will automatically use GPU if available (`torch.cuda.is_available()`), otherwise it runs on CPU.

---

## About `build.sh`

`build.sh` is a convenience wrapper around `docker build`:

```bash
#!/bin/bash
OPERATOR=${OPERATOR:-cpu}
docker build \
  -t pyannote-${OPERATOR} \
  --build-arg HF_TOKEN=${HF_TOKEN} \
  --build-arg OPERATOR=${OPERATOR} \
  --platform linux/amd64 .
```

- **`OPERATOR`** selects the base stage in the `Dockerfile`:
  - `cpu` -> `FROM python:3.10-slim`
  - `nvidia` -> `FROM cupy/cupy:v13.2.0`
- **`HF_TOKEN`** is forwarded into the build to pre-download the model via `model_cache.sh`.
- `--platform linux/amd64` ensures compatibility when building from Apple Silicon.

You can either call `./build.sh` (CPU default) or override `OPERATOR`:

```bash
# CPU
export HF_TOKEN=hf_your_token
./build.sh

# NVIDIA
export HF_TOKEN=hf_your_token
OPERATOR=nvidia ./build.sh
```

---

## Build without the script

```bash
# CPU image
docker build -t pyannote-cpu \
  --build-arg HF_TOKEN=${HF_TOKEN} \
  --build-arg OPERATOR=cpu \
  --platform linux/amd64 .

# NVIDIA image
docker build -t pyannote-nvidia \
  --build-arg HF_TOKEN=${HF_TOKEN} \
  --build-arg OPERATOR=nvidia \
  --platform linux/amd64 .
```

---

## Run and test

Start the container (CPU example):
```bash
docker run --rm -p 8080:8080 pyannote-cpu
```

Health check:
```bash
curl -s http://localhost:8080/health-check
```

Diarize via JSON URL:
```bash
curl -s -X POST http://localhost:8080/diarize \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://your-bucket/audio.wav"}' | jq .
```

Diarize via multipart file upload:
```bash
curl -s -X POST http://localhost:8080/diarize \
  -F file=@/path/to/audio.wav | jq .
```

Diarize via raw bytes:
```bash
curl -s -X POST http://localhost:8080/diarize \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @/path/to/audio.wav | jq .
```

---

## Implementation notes

- The service is implemented with FastAPI (`uvicorn` entrypoint) in `app/main.py`.
- The model pipeline is created in `app/model.py` and moved to GPU automatically if available.
- `model_cache.sh` uses `HF_TOKEN` at build time to pre-download the `pyannote/speaker-diarization-3.1` pipeline to the container cache.
- Supported audio types are those readable by pyannote/torchaudio/ffmpeg via the pipeline; common formats like WAV typically work best.

---

## Troubleshooting

- If build fails with missing token, ensure `HF_TOKEN` is exported in your shell.
- If running on GPU, ensure `docker run --gpus all` and the NVIDIA toolkit are installed.
- If you build on Apple Silicon for a Linux deployment, keep `--platform linux/amd64` in the build.


