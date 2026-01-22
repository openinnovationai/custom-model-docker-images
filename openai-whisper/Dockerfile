ARG OPERATOR=nvidia

FROM nvidia/cuda:13.1.1-runtime-ubuntu24.04 AS nvidia
FROM python:3.10-slim AS cpu

FROM ${OPERATOR}

ENV HOME=/home/runner
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="${VIRTUAL_ENV}/bin:$HOME/.local/bin:$PATH"

SHELL [ "/bin/bash", "-c" ]

RUN <<eot
    set -ex

    apt update 
    apt install ffmpeg libsndfile1 -y
    rm -rf /var/lib/apt/lists/*
    
    useradd --uid 10000 --create-home runner
    mkdir -p /app && chown runner /app
    mkdir -p /home/runner/.local/share/uv && chown -R runner:runner /home/runner
    passwd -l root
eot

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx $HOME/.local/bin/

USER runner

WORKDIR /app

RUN uv venv ${VIRTUAL_ENV}

RUN --mount=type=bind,source=uv.lock,target=uv.lock \
  --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  uv sync --frozen --no-install-project --no-dev

RUN mkdir -p /home/runner/.cache/whisper
COPY --chown=runner:runner ./large-v3-turbo.pt /home/runner/.cache/whisper/large-v3-turbo.pt
COPY --chown=runner:runner server.py /app/

EXPOSE 8080

CMD [ "uv", "run", "python", "server.py" ]
