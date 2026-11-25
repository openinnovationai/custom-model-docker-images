# syntax=docker/dockerfile:1.7
#
# EngineFabric custom model image - template.
#   CPU : docker buildx build --platform linux/amd64 -t my-model:cpu --load .
#   GPU : docker buildx build --platform linux/amd64 \
#           --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu24.04 \
#           -t my-model:cu128 --load .      (and point torch at the cu128 index in pyproject.toml)
# Pin BASE_IMAGE by digest (@sha256:...) for release builds.

ARG BASE_IMAGE=python:3.12-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.22

FROM ${UV_IMAGE} AS uv

# ---------------------------------------------------------------- builder
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/app/.venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /app
# Dependencies first: this layer is reused until pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv python install 3.12 \
    && uv sync --frozen --no-dev --no-install-project

# Weights: pinned repo + commit SHA. Own layer, so code edits don't re-download.
ARG MODEL_ID=distilbert/distilbert-base-uncased-finetuned-sst-2-english
ARG MODEL_REVISION=714eb0fa89d2f80546fda750413ed43d93601a13
ENV MODEL_ID=${MODEL_ID} MODEL_REVISION=${MODEL_REVISION} MODEL_DIR=/models/model
COPY scripts/preload.py scripts/preload.py
# Gated model? pass the token as a build secret, never as ARG/ENV:
#   RUN --mount=type=secret,id=hf_token,env=HF_TOKEN /app/.venv/bin/python scripts/preload.py
RUN /app/.venv/bin/python scripts/preload.py

# ---------------------------------------------------------------- runtime
FROM ${BASE_IMAGE} AS runtime

ENV HOME=/home/runner \
    PATH=/app/.venv/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/models/model \
    HF_HOME=/tmp/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN groupadd --gid 10000 runner \
    && useradd --uid 10000 --gid 10000 --create-home --shell /usr/sbin/nologin runner \
    && mkdir -p /app /models \
    && chown 10000:10000 /app /models \
    && passwd -l root

# --chown on COPY: a later `RUN chown -R` would duplicate these multi-GB layers.
COPY --from=builder /opt/python /opt/python
COPY --from=builder --chown=10000:10000 /app/.venv /app/.venv
COPY --from=builder --chown=10000:10000 /models /models
COPY --chown=10000:10000 server.py /app/server.py

WORKDIR /app
USER 10000:10000

# Prove the image is air-gapped: load the model with networking disabled.
RUN --network=none MODEL_DEVICE=cpu python -c "import server; server.load_model()"

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health-check', timeout=4)"]

# One worker per container: each worker would load its own copy of the weights.
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
