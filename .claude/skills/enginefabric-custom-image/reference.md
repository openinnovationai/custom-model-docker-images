# EngineFabric custom image: patterns beyond the FastAPI template

## Which repo example to start from

| Model type | Copy from | Why |
|---|---|---|
| Any Python model, GPU, production-grade | `openai-privacy-filter/` | Reference quality: digest-pinned CUDA base, multi-stage uv, pinned HF revision preload, fail-closed device, busy → 503, tests with fake model |
| Speech-to-text, OpenAI-compatible | `omniasr-server/` | `/v1/audio/transcriptions` + `/v1/models` mirroring OpenAI, error handlers, non-HF (fairseq2) preload via `scripts/preload.py` |
| CPU **and** GPU variants | `speaker-diarization-3.1/` (two Dockerfiles), `openai-whisper/` (`ARG OPERATOR` stage switch) | Build-time choice of base |
| Weights from Hugging Face cache layout, Ray Serve | `gliner/gliner/serve/Containerfile` | Bundled HF cache outside `$HOME`, entrypoint re-exports cache vars, `RUN --network=none` offline proof |
| LLM / VLM via vLLM | `deepseek-ocr/` | vLLM on 127.0.0.1:8081 behind nginx on 8080, `/health-check` proxied to `/v1/models` |
| LitServe | `vibe-voice/` | `ls.LitServer(api, healthcheck_path="/health-check").run(port=8080)` |
| Image generation/editing | `qwen-image-edit-image/` | OpenAI-images-shaped `/v1/images/edits` with base64 output |

Older examples (deepfake-detector, qwen-image-edit, openai-whisper, vibe-voice) predate some rules: `RUN chown -R` after `COPY . .`, `uv:latest`, `@app.on_event`, `uv run` as CMD, weights loaded by name at runtime. Copy their API shape, not their Dockerfile.

## Wrapping an existing HTTP server (vLLM, TGI, Triton, SGLang)

These servers already serve the model but don't expose `/health-check`. Options, simplest first:

1. **The server lets you choose the port and has its own health route:** run it on `0.0.0.0:8080` and add a tiny route alias if it supports one. Otherwise:
2. **nginx in front** (as in `deepseek-ocr/`): the engine listens on `127.0.0.1:8081`, nginx on `8080` with
   ```nginx
   location /health-check { proxy_pass http://127.0.0.1:8081/v1/models; }   # or /health
   location /             { proxy_pass http://127.0.0.1:8081; proxy_read_timeout 600s; }
   ```
   As non-root, nginx needs every writable path moved into a 10000-owned dir: `pid`, `error_log`, `access_log`, `client_body_temp_path`, `proxy_temp_path`, `fastcgi_temp_path`, `scgi_temp_path`, `uwsgi_temp_path`. Use `daemon off;` and `listen 8080;` (ports < 1024 need root).
3. **Supervise both processes** in the entrypoint so the container dies if either dies:
   ```bash
   #!/bin/bash
   python3 -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port 8081 --model /models/model --served-model-name <name> &
   nginx -c /home/runner/nginx.conf &
   wait -n            # first child to exit ends the container -> platform restarts it
   exit $?
   ```
   Use `exec` when there is only one process so signals reach it.

For vLLM images, copy the weights into the image (`COPY` a pre-downloaded dir or a preload stage) and pass `--model /models/...`, never a hub id.

## LitServe

```python
server = ls.LitServer(api, healthcheck_path="/health-check")
server.run(port=8080)          # binds 0.0.0.0
```
LitServe's health returns 200 only after `setup()` finishes, so put model loading in `setup()`.

## Ray Serve

`deployment-config.yaml`: `http_options: {host: 0.0.0.0, port: 8080}`. Ray Serve has no `/health-check` route by default. Add one in the ingress deployment that returns 200 once the model replica is up. Ray writes to `/tmp/ray`, so the dir must be writable by 10000.

## CPU and GPU from one Dockerfile

Either parameterise the base (`ARG BASE_IMAGE`, as in `template/`), or select a named stage:

```dockerfile
ARG OPERATOR=nvidia
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS nvidia
FROM python:3.12-slim-bookworm AS cpu
FROM ${OPERATOR}
```
Torch wheels differ per target. The CPU index (`/whl/cpu`) keeps CPU images around 1 GB smaller. CUDA wheels (`/whl/cu128`) bundle the CUDA runtime, so `nvidia/cuda:*-runtime` (or even `-base`) plus the host driver is enough. You only need `-cudnn-` or `-devel` when a dependency links against them or compiles kernels.

When the base image has no Python (nvidia/cuda), either `apt-get install python3 python3-venv` and set `UV_PYTHON_DOWNLOADS=never`, or let uv install a managed Python into `/opt/python` in the builder and copy `/opt/python` into the runtime (template approach; same Python on every base).

## Weights that are too large or change often

Baking weights in is the default: startup is reproducible and air-gapped. If an image would be impractically large (tens of GB), check with the platform team before switching to mounted model storage. Never fall back to downloading at startup.

Build-only sanity build without weights (fast iteration): add `ARG PRELOAD_MODEL=1` and skip `preload.py` when it is `0`, as `openai-privacy-filter` does. Tag such images `-no-model` so they never get deployed.

## Exporting an image for an air-gapped registry

```bash
docker image save --output dist/<name>-amd64.tar <name>:<tag>      # then load/push on the other side
```
Keep `dist/` and `*.tar` in `.dockerignore` and `.gitignore`.
