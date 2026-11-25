# <model-name>

EngineFabric custom image serving `<MODEL_ID>` on port `8080`, running as UID 10000, with weights baked in at a pinned revision (no network needed at startup).

## Build

```bash
make lock          # after editing pyproject.toml
make test          # contract tests with a fake model
make build         # linux/amd64 CPU image
make build-gpu     # linux/amd64 CUDA image (switch the torch index in pyproject.toml first)
make offline-smoke # run with --network none and call health + predict
```

## Run

```bash
docker run --rm -p 8080:8080 my-model:cpu
docker run --rm --gpus all -p 8080:8080 my-model:gpu
```

## Endpoints

### `GET /health-check`

```json
{"status": "healthy", "message": "Model is ready and healthy"}
```

Returns 503 while the model is loading.

### `POST /v1/classify`

```bash
curl localhost:8080/v1/classify -H 'Content-Type: application/json' \
  -d '{"instances": ["great product", "terrible support"]}'
```

```json
{"labels": [{"label": "POSITIVE", "score": 0.9999}, {"label": "NEGATIVE", "score": 0.9994}]}
```

At most 64 instances per request. Requests are serialized per container.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_DEVICE` | auto | `cuda` or `cpu`. `cuda` fails startup if no GPU is visible |
| `MODEL_DIR` | `/models/model` | Baked-in weights directory |

## Build args

| Arg | Purpose |
| --- | --- |
| `BASE_IMAGE` | `python:3.12-slim-bookworm` (CPU) or an `nvidia/cuda:*-runtime-*` image |
| `MODEL_ID`, `MODEL_REVISION` | Hugging Face repo and commit SHA to bake in |
