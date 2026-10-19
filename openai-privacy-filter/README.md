# OpenAI Privacy Filter

GPU-backed FastAPI service for [OpenAI Privacy Filter](https://github.com/openai/privacy-filter), built for the OICM custom-model runtime contract and NVIDIA B300 GPUs.

The production image uses CUDA 13.0 and PyTorch 2.13, runs as a non-root user, listens on port `8080`, and includes the pinned Privacy Filter checkpoint so startup does not require network access. The model is warmed before the health endpoint becomes available.

## Build

Build the deployable `linux/amd64` image, including the checkpoint:

```bash
docker build --platform linux/amd64 -t openai-privacy-filter:cu130 .
```

For a fast source and dependency build used only to validate the image layers, disable preloading:

```bash
docker build --platform linux/amd64 \
  --build-arg PRELOAD_MODEL=0 \
  -t openai-privacy-filter:cu130-no-model .
```

That validation image has no checkpoint and is not deployment-ready. Use the default build for OICM.

The default checkpoint is pinned to a Hugging Face revision. Override it deliberately with `--build-arg OPF_MODEL_REVISION=<commit>`.

## Run on NVIDIA GPU

The host needs a B300-compatible NVIDIA driver and NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all -p 8080:8080 openai-privacy-filter:cu130
```

The container sets `OPF_DEVICE=cuda`, so it fails startup instead of silently using the CPU when GPU access is unavailable.

## Endpoints

### Health check

```bash
curl http://localhost:8080/health-check
```

Example response:

```json
{
  "status": "ok",
  "device": "cuda",
  "device_name": "NVIDIA B300"
}
```

`GET /health` is an alias for integrations that already use that path.

### Redact text

```bash
curl http://localhost:8080/v1/redact \
  -H 'Content-Type: application/json' \
  -d '{"text":"Contact Alice at alice@example.com."}'
```

The response contains the complete `redacted_text`, detected-span metadata, summary, and schema version. Original text values are omitted by default. Set `include_original` to `true` only in a trusted environment that explicitly needs them.

Redact only selected categories by passing `labels`:

```bash
curl http://localhost:8080/v1/redact \
  -H 'Content-Type: application/json' \
  -d '{
    "text":"Alice: alice@example.com, +1 202-555-0147",
    "labels":["private_email","private_phone"]
  }'
```

Allowed labels are `private_person`, `private_address`, `private_email`, `private_phone`, `private_url`, `private_date`, `account_number`, and `secret`. Omitting `labels` or using `null` redacts every category.

Requests are limited to 100,000 characters. Inference is serialized per container; a concurrent request receives HTTP `503` with `Retry-After: 1`. Run one Uvicorn worker per model instance to avoid loading duplicate model weights.

## Test

The tests use a fake redactor and do not download the checkpoint:

```bash
uv sync
uv run pytest
```

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPF_DEVICE` | `cuda` | Inference device. Production accepts `cuda` or `cpu`. |
| `OPF_CHECKPOINT` | `/models/privacy-filter` | Checkpoint directory. |
| `OPF_TORCH_COMPILE` | unset | Set to `1` to test PyTorch compilation; benchmark before enabling. |
| `OPF_TORCH_COMPILE_MODE` | `default` | Compilation mode when `OPF_TORCH_COMPILE=1`. |

Privacy Filter detects and masks sensitive text but cannot guarantee that every identifier is found. Applications should define their own retention and fail-closed policies around this service.
