# Select operator at build time: amd | nvidia | cpu
ARG OPERATOR=cpu

# Base stages for each operator
FROM registry.gitlab.com/openinnovationai/platform/mlops/mlops-serving/rocm-6.1-pytorch-2.1.2-flash-attn AS base-amd
FROM cupy/cupy:latest AS base-nvidia
FROM python:3.11 AS base-cpu

# Common runtime stage selected by OPERATOR
FROM base-${OPERATOR} AS runtime

USER root

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

EXPOSE 8080-9000

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

# Ensure required directories exist
RUN mkdir -p /var/log/supervisor /app

# Install appropriate PyTorch per operator first
ARG OPERATOR
RUN echo "Building with OPERATOR: ${OPERATOR}"
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel

RUN --mount=type=cache,target=/root/.cache/pip \
    bash -lc 'set -euo pipefail; \
    echo "Installing PyTorch for operator: ${OPERATOR}"; \
    if [ "${OPERATOR}" = "amd" ]; then \
      python3 -m pip install --no-cache-dir --pre torch==2.5.0.dev20240726 torchvision==0.20.0.dev20240726 --index-url https://download.pytorch.org/whl/nightly/rocm6.1; \
    elif [ "${OPERATOR}" = "nvidia" ]; then \
      python3 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121; \
    else \
      python3 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    fi'

# Python deps (install after PyTorch to avoid conflicts)
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r /app/requirements.txt

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
