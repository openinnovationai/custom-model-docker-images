# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE=nvidia/cuda:13.0.2-runtime-ubuntu24.04@sha256:6a0e31b59e70890446f2c17356d6efc0d54260090a8c63d6ca7c2ad049db95d2

FROM ${CUDA_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:/usr/local/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.2 /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

ARG OPF_MODEL_REVISION=7ffa9a043d54d1be65afb281eddf0ffbe629385b
ARG PRELOAD_MODEL=1
ENV OPF_CHECKPOINT=/models/privacy-filter \
    OPF_MODEL_REVISION=${OPF_MODEL_REVISION}

COPY scripts/preload.py scripts/preload.py
RUN mkdir -p /models \
    && if [ "${PRELOAD_MODEL}" = "1" ]; then python scripts/preload.py; fi


FROM ${CUDA_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/runner \
    OPF_CHECKPOINT=/models/privacy-filter \
    OPF_DEVICE=cuda \
    PATH=/app/.venv/bin:${PATH} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 python3 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10000 runner \
    && useradd --uid 10000 --gid 10000 --create-home --shell /usr/sbin/nologin runner \
    && mkdir -p /app /models \
    && chown -R runner:runner /app /models \
    && passwd -l root

WORKDIR /app
COPY --from=builder --chown=runner:runner /app/.venv /app/.venv
COPY --from=builder --chown=runner:runner /models /models
COPY --chown=runner:runner server.py ./

USER runner:runner

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health-check', timeout=3)"]

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-access-log"]
