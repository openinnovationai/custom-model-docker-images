## Custom models docker images for OICM platform

This repo contains examples for the custom docker images deployment.

### Available Services

#### Speaker Diarization 3.1
Location: `speaker-diarization-3.1/`

Endpoints:
- `GET /health-check` - Health check endpoint
- `POST /v1/audio/diarization` - Speaker diarization endpoint (supports JSON URL, multipart file upload, or raw bytes)

See [speaker-diarization-3.1/README.md](speaker-diarization-3.1/README.md) for detailed documentation.
