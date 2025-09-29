ARG OPERATOR

FROM cupy/cupy:v13.2.0 AS nvidia
FROM python:3.10-slim AS cpu

FROM ${OPERATOR}

EXPOSE 8080

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN} \
    HF_HOME=/opt/huggingface \
    TRANSFORMERS_CACHE=/opt/huggingface/transformers \
    XDG_CACHE_HOME=/opt/.cache \
    MPLCONFIGDIR=/opt/.cache/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x /app/model_cache.sh 

# cache the model if you want to run it in airgapped environment
RUN /app/model_cache.sh

RUN useradd --uid 10000 runner

RUN mkdir -p /home/runner /app \
    && chown -R runner /home/runner /app /opt/huggingface /opt/.cache \
    && passwd -l root

RUN chown -R runner:runner /app

USER runner:runner

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

