ARG OPERATOR

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu2 AS nvidia
FROM python:3.12-bookworm AS cpu

FROM ${OPERATOR}

ENV HOME=/home/runner
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_NO_CACHE=1

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="${VIRTUAL_ENV}/bin:$HOME/.local/bin:$PATH"


SHELL [ "/bin/bash", "-c" ]

RUN <<eot
    set -ex
    # Security: add user runner
    useradd --uid 10000 runner

    mkdir /home/runner && chown runner /home/runner
    mkdir -p /app && chown runner /app

    # Security: disable root login
    passwd -l root
eot

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx $HOME/.local/bin/

USER runner

WORKDIR /app

RUN uv venv ${VIRTUAL_ENV}

RUN --mount=type=bind,source=uv.lock,target=uv.lock \
  --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  uv sync --frozen --no-install-project --no-dev --active

COPY --chown=runner:runner vibevoice /app/vibevoice

RUN uv pip install /app/vibevoice
COPY --chown=runner:runner voices /app/voices
COPY --chown=runner:runner models /app/models
COPY --chown=runner:runner server.py /app/

CMD [ "uv", "run", "server.py" ]
