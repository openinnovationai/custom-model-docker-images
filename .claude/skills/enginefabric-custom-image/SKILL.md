---
name: enginefabric-custom-image
description: Use when packaging a model (Hugging Face, PyTorch, ASR/TTS, OCR, NER, diffusion, vLLM, etc.) as a custom Docker image to deploy on EngineFabric / OICM, when adding a new example to the custom-model-docker-images repo, or when an EngineFabric custom-image deployment never becomes Ready, fails its health check, crashes on permissions, or tries to download weights at startup.
---

# EngineFabric Custom Model Image

EngineFabric (OICM) runs a custom image as-is and treats it as ready only when its health endpoint answers. The image must meet a fixed runtime contract, bundle its own weights and need no network at startup. Everything below comes from the examples in this repo; `template/` is a working reference image (built, run air-gapped and tested).

## The contract (non-negotiable)

| Requirement | Value |
|---|---|
| Architecture | `linux/amd64`. Always pass `--platform linux/amd64`; an Apple-silicon build is arm64 by default and will not start on the cluster |
| User | non-root `runner`, **UID/GID 10000**: `USER 10000:10000` (or `USER runner` with uid 10000) |
| Port | **8080**, bound to `0.0.0.0` (not `127.0.0.1`), `EXPOSE 8080` |
| Health | `GET /health-check` → **200** only when the model is loaded and can serve. Deployment waits for it. Return 503 (or don't listen yet) while loading. Preferred body: `{"status": "healthy", "message": "Model is ready and healthy"}` |
| Inference | Any route you like. Name it after the task (`/v1/redact`, `/v1/audio/transcriptions`, `/v1/images/edits`). If an established API exists for the task (OpenAI audio/images/chat), mirror it so existing clients work |
| Weights | Inside the image, at a pinned revision. Startup makes **zero** network calls |
| Process | One model copy per container: `--workers 1`, concurrency handled in-process |

Anything the process reads must be readable by UID 10000; anything it writes (caches, temp files, logs, nginx pid) must be in a directory owned by 10000.

## Workflow

1. **Copy the template** to `<model-name>/` at the repo root: `cp -r .claude/skills/enginefabric-custom-image/template <model-name>`.
2. **Dependencies**: edit `pyproject.toml` with exact `==` pins, then `uv lock`. GPU: pick the torch index matching the CUDA base (`cu126`, `cu128`, `cu130`) and the GPU (Blackwell B200/B300 needs CUDA ≥ 12.8 builds).
3. **Weights**: set `MODEL_ID` and `MODEL_REVISION` (a commit SHA from `https://huggingface.co/api/models/<id>` → `sha`, never `main`). Narrow `allow_patterns` in `scripts/preload.py` to the files the runtime loads. Non-HF sources (fairseq2, torch hub, S3): write the equivalent download into `preload.py` and point the library's cache env var at a directory inside the image.
4. **Server**: replace `load_model()` and the `/v1/classify` route in `server.py`. Keep the lifespan load, warm-up call, `/health-check` and the lock. Keep `create_app(loader=...)` so tests can inject a fake.
5. **Tests**: update `tests/test_server.py`; `make test` must pass with no weights and no GPU.
6. **Build and prove it**: `make build IMAGE=<name>` then `make offline-smoke IMAGE=<name>` (adapt the predict payload). GPU image: `make build-gpu`, then run with `--gpus all` on a GPU host.
7. **Document**: `<model-name>/README.md` with build, run, every endpoint with a `curl` example and its response, environment-variable table, hardware notes. Add the service to the root `README.md` "Available Services" list.
8. **Push the image** to the registry EngineFabric pulls from, with an explicit tag (`<model>:cu128-<model-version>`), never only `latest`.

## Dockerfile rules

The template `Dockerfile` already applies all of these. When you change it or start from another base, keep them.

- **Multi-stage.** Builder installs deps and downloads weights; runtime copies only `/app/.venv`, `/models` and the code. No compilers, git or uv-cache in the final image.
- **Layer order for caching:** `pyproject.toml` + `uv.lock` → `uv sync --frozen --no-dev --no-install-project` → weights → source code last. Code edits must not re-download weights.
- **Pin everything:** base image tag (digest for releases), uv image (`ghcr.io/astral-sh/uv:0.8.22`, never `:latest`), Python packages via `uv.lock` + `--frozen`, model revision SHA.
- **`COPY --chown=10000:10000`** for venv, weights and code. A later `RUN chown -R` copies the whole multi-GB layer again.
- **Create the user explicitly:** `groupadd --gid 10000 runner && useradd --uid 10000 --gid 10000 --create-home --shell /usr/sbin/nologin runner`, then `passwd -l root`.
- **Offline at runtime:** set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and load from a local path. Prove it in the build: `RUN --network=none python -c "import server; server.load_model()"` (force `MODEL_DEVICE=cpu` there, since builders have no GPU).
- **Keep weights outside `$HOME`** (`/models/...` or `/opt/<name>/...`). The platform can mount over or change `HOME`, which hides anything cached there.
- **Start the server directly:** `CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]` with the venv on `PATH`. Not `uv run` (it can re-sync and reach the network) and not a shell wrapper unless you need two processes.
- **`HEALTHCHECK` via Python `urllib`.** Slim images have no `curl`. Give `--start-period` enough time for the model to load.
- **`.dockerignore`** must exclude `.git`, `.env*`, credentials, `.venv`, `__pycache__`, local `models/`, tests and tarballs. Never `COPY . .` without one.
- **Gated or private weights:** `RUN --mount=type=secret,id=hf_token,env=HF_TOKEN ...`. Never `ARG`/`ENV HF_TOKEN`: it stays in the image history.

## Server rules

- Load in the FastAPI `lifespan` (not deprecated `@app.on_event`, not lazily on the first request). Uvicorn doesn't accept connections until lifespan startup returns, so health can't pass before the model is ready.
- Run one warm-up inference before `yield` so the first user request doesn't pay compile/allocation cost.
- Use plain `def` routes for blocking inference so they run in the threadpool and `/health-check` stays responsive. In `async def`, use `await asyncio.to_thread(...)`.
- Guard the single model with a `threading.Lock`. To shed load instead of queueing, use `lock.acquire(blocking=False)` → 503 + `Retry-After: 1`.
- **Fail closed on device:** if `MODEL_DEVICE=cuda` and no GPU is visible, raise at startup. Never fall back to CPU silently.
- Validate input with Pydantic (`extra="forbid"`, max lengths and batch sizes). Return 4xx for bad input, 500 with a generic message for inference errors, and log the traceback.
- Clean up temp files (uploads) in `finally`.

## Other serving stacks

Not every model is a FastAPI app. See [reference.md](reference.md) for: wrapping vLLM or another OpenAI-compatible server behind nginx so `/health-check` exists; LitServe (`healthcheck_path="/health-check"`); Ray Serve (`http_options.port: 8080`); CPU/GPU variants from one Dockerfile; and which repo example to copy for each model type.

## Verification before calling it done

Run each one and read the output. Don't assume.

| Check | Command | Expect |
|---|---|---|
| Lint | `docker buildx build --check .` | no warnings |
| Unit | `make test` | all pass |
| Arch / user / port | `docker inspect <img> --format '{{.Architecture}} {{.Config.User}} {{.Config.ExposedPorts}}'` | `amd64 10000:10000 map[8080/tcp:{}]` |
| Air-gapped start | `make offline-smoke` (`--network none`) | `uid=10000(runner)`, health 200, real prediction |
| GPU (if GPU image) | `docker run --gpus all -p 8080:8080 <img>` then `curl localhost:8080/health-check` | 200, and logs show `cuda` |
| Size | `docker images <img>` | no duplicate weight layers, no stray checkpoints |

If the deployment then fails on EngineFabric, use the `ef-workload-debug` skill to read pod events and logs.

## Common mistakes

| Symptom on the platform | Cause | Fix |
|---|---|---|
| `exec format error` / CrashLoop right away | arm64 image built on a Mac | `--platform linux/amd64` |
| Never becomes Ready | No `/health-check`, wrong port, bound to `127.0.0.1`, or `--start-period`/platform probe shorter than load time | Fix the route and port; bind `0.0.0.0` |
| Ready, but first requests 500 | Health returns 200 before the model is loaded (lazy loading) | Load in lifespan; 503 until loaded |
| `PermissionError` under `/app`, `~/.cache`, nginx `pid`/temp paths | Files owned by root, runtime is UID 10000 | `COPY --chown=10000:10000`; point caches/temp/pid at 10000-owned dirs |
| Hangs or `ConnectionError` at startup in the cluster | Hidden download (HF hub, torch hub, fairseq2, `whisper.load_model("turbo")` by name) | Bake weights in; set offline env vars; load by local path; `--network none` test |
| Weights "missing" though they're in the image | Cached under `$HOME`, which the platform overrode | Store in `/models` or `/opt/...`, set cache env explicitly |
| OOM / duplicate weights in GPU memory | `--workers N` loads N copies | One worker; scale with replicas |
| GPU deployment silently slow | Torch CPU wheel, or CPU fallback | CUDA index matching the base image; fail closed on `cuda` |
| Builds differ day to day | `:latest` bases, unpinned deps, model revision `main` | Pin tags, `uv.lock --frozen`, SHA revision |
