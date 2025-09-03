## syntax=docker/dockerfile:1.6
# Select operator at build time: amd | nvidia | cpu
ARG OPERATOR=cpu

# Base stages for each operator
FROM registry.gitlab.com/openinnovationai/platform/mlops/mlops-serving/rocm-6.1-pytorch-2.1.2-flash-attn AS base-amd
FROM cupy/cupy:latest AS base-nvidia
FROM python:3.11-slim AS base-cpu

# Common runtime stage selected by OPERATOR
FROM base-${OPERATOR} AS runtime

USER root

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

EXPOSE 8080

WORKDIR /app

# System dependencies (only what's needed)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    nginx \
    supervisor \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --system --create-home --home-dir /home/appuser --shell /usr/sbin/nologin appuser \
    && mkdir -p /var/log/supervisor /app \
    && chown -R appuser:appuser /app /var/log/supervisor

# Python deps
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -r /app/requirements.txt

# Install appropriate PyTorch per operator
ARG OPERATOR
RUN --mount=type=cache,target=/root/.cache/pip \
    bash -lc 'set -euo pipefail; \
    if [ "${OPERATOR}" = "amd" ]; then \
    python3 -m pip uninstall -y torch torchvision || true; \
    python3 -m pip install --no-cache-dir --pre torch==2.5.0.dev20240726 torchvision==0.20.0.dev20240726 --index-url https://download.pytorch.org/whl/nightly/rocm6.1; \
    elif [ "${OPERATOR}" = "nvidia" ]; then \
    python3 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121; \
    else \
    python3 -m pip install --no-cache-dir torch torchvision; \
    fi'

# Nginx and supervisor config
COPY proxy.conf /etc/nginx/conf.d/proxy.conf
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# App code and runtime config
COPY app/ /app/
COPY deployment-config.yaml /app/deployment-config.yaml
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Keep root to run nginx/supervisord reliably
USER root

ENV SERVER_START_CMD=/usr/bin/supervisord
ENTRYPOINT ["/usr/bin/supervisord"]