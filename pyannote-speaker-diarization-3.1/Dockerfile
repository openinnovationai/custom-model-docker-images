## syntax=docker/dockerfile:1.6
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

# Hugging Face build-time token and cache locations
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN} \
    HF_HOME=/opt/huggingface \
    TRANSFORMERS_CACHE=/opt/huggingface/transformers \
    XDG_CACHE_HOME=/opt/.cache \
    MPLCONFIGDIR=/opt/.cache/matplotlib

RUN mkdir -p /opt/huggingface /opt/.cache /opt/.cache/matplotlib

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

# Allow nginx to create request body temp files when running as non-root
RUN mkdir -p /var/lib/nginx/body \
    && chown -R 10000 /var/lib/nginx

# Allow nginx to write logs as non-root
RUN mkdir -p /var/log/nginx \
    && chown -R 10000 /var/log/nginx

# Create non-root user (runner: UID 10000) and minimal dirs (per reference)
RUN useradd --uid 10000 runner \
    && mkdir -p /home/runner /app \
    && chown -R runner /home/runner /app /opt/huggingface /opt/.cache \
    && passwd -l root

# Install appropriate PyTorch per operator first
ARG OPERATOR
RUN echo "Building with OPERATOR: ${OPERATOR}"
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel

RUN --mount=type=cache,target=/root/.cache/pip \
    bash -lc 'set -euo pipefail; \
    echo "Installing PyTorch for operator: ${OPERATOR}"; \
    if [ "${OPERATOR}" = "amd" ]; then \
    python3 -m pip install --no-cache-dir --pre torch==2.5.0.dev20240726 torchvision==0.20.0.dev20240726 torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.1; \
    elif [ "${OPERATOR}" = "nvidia" ]; then \
    python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124; \
    python3 -m pip install flash-attn==2.7.3 --no-build-isolation; \
    else \
    python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi'

# Python deps (install after PyTorch to avoid conflicts)
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r /app/requirements.txt

# Pre-download gated model at build time using HF_TOKEN via script
ENV PYANNOTE_MODEL_ID=pyannote/speaker-diarization-3.1
COPY prefetch_model.sh /app/prefetch_model.sh
RUN chmod +x /app/prefetch_model.sh && /app/prefetch_model.sh

# Nginx and supervisor config
COPY nginx.conf /etc/nginx/nginx.conf
COPY proxy.conf /etc/nginx/conf.d/proxy.conf
COPY supervisord.conf /etc/supervisor/supervisord.conf

# App code and runtime config
COPY app/ /app/
COPY deployment-config.yaml /app/deployment-config.yaml
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Drop to non-root for runtime
USER runner

# Runtime environment for runner
ENV HOME=/home/runner \
    PATH="$PATH:/home/runner/.local/bin"

ENV SERVER_START_CMD=/usr/bin/supervisord
ENTRYPOINT ["/usr/bin/supervisord"]