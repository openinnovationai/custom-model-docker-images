ARG OPERATOR=nvidia

# --- Builder Stage ---
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04 AS builder-nvidia
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates

FROM python:3.10-slim AS builder-cpu

FROM builder-${OPERATOR} AS builder

ENV HOME=/home/runner
ENV PATH="$PATH:$HOME/.local/bin"
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_NO_CACHE=1

ENV VIRTUAL_ENV=/deps/.venv
ENV UV_PYTHON_INSTALL_DIR=/deps/python
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

SHELL [ "/bin/bash", "-c" ]

RUN <<eot
    set -ex
    useradd --uid 10000 -m runner || true
    mkdir -p /deps /home/runner/.local/bin /home/runner/.local/share/uv
    chown -R runner:runner /home/runner /deps
    passwd -l root
eot

COPY --chown=runner:runner --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx $HOME/.local/bin/

USER runner

RUN <<eot
    set -ex
    uv venv ${VIRTUAL_ENV} --python 3.10
eot

RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --active


# --- Runtime Bases ---
FROM nvidia/cuda:13.1.1-runtime-ubuntu24.04 AS runtime-nvidia
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

FROM python:3.10-slim AS runtime-cpu


# --- Final Stage ---
FROM runtime-${OPERATOR}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/runner
ENV PATH="$PATH:$HOME/.local/bin"
ENV VIRTUAL_ENV=/deps/.venv
ENV UV_PYTHON_INSTALL_DIR=/deps/python
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
ENV UV_NO_CACHE=1

SHELL [ "/bin/bash", "-c" ]

RUN <<eot
    set -ex

    apt-get update -y
    apt-get clean
    rm -rf /var/lib/apt/lists/*

    # Security: add user runner
    useradd --uid 10000 -m runner || true
    chown -R runner /home/runner/

    # Virtual environment directory
    mkdir -p /deps
    chown runner /deps

    # Create app directory
    mkdir -p /app
    chown runner /app

    # Security: disable root login
    passwd -l root
eot

USER runner

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx $HOME/.local/bin/

# Copy virtual environment and managed python from builder
COPY --chown=runner:runner --from=builder /deps /deps

# Application configuration
WORKDIR /app
ENV OICM_MODEL_PATH=/app/model/mxbai-rerank-large-v1

# Copy model and server script
COPY --chown=runner:runner ./models/mxbai-rerank-large-v1 /app/model/mxbai-rerank-large-v1
COPY --chown=runner:runner server.py /app/

EXPOSE 8080

CMD ["python", "server.py"]