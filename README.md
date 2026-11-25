## Custom models docker images for OICM platform

This repo contains examples for the custom docker images deployment.

### Building a new image (humans and AI agents)

Read [`.claude/skills/enginefabric-custom-image/SKILL.md`](.claude/skills/enginefabric-custom-image/SKILL.md) first. It covers the EngineFabric runtime contract (linux/amd64, UID 10000, port 8080, `GET /health-check`, weights baked in, no network at startup), Dockerfile and server best practices, and a verification checklist. It also ships a tested starting point in [`template/`](.claude/skills/enginefabric-custom-image/template/).

Claude Code picks the skill up automatically inside this repo. Other agents can be pointed at the `SKILL.md` file directly.

### Available Services

#### Speaker Diarization 3.1
Location: `speaker-diarization-3.1/`

Endpoints:
- `GET /health-check` - Health check endpoint
- `POST /v1/audio/diarization` - Speaker diarization endpoint (supports JSON URL, multipart file upload, or raw bytes)

See [speaker-diarization-3.1/README.md](speaker-diarization-3.1/README.md) for detailed documentation.

#### OpenAI Privacy Filter
Location: `openai-privacy-filter/`

Endpoints:
- `GET /health-check` - Health check with the active inference device
- `POST /v1/redact` - Detect and mask selected privacy categories in text

See [openai-privacy-filter/README.md](openai-privacy-filter/README.md) for B300 build and deployment instructions.
