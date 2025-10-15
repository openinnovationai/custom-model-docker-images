ARG OPERATOR

FROM cupy/cupy:v13.2.0 AS nvidia
FROM python:3.10-slim AS cpu

FROM ${OPERATOR}

EXPOSE 8080

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ENV HF_HOME=/app
ENV TORCH_HOME=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --uid 10000 runner

RUN mkdir -p /home/runner /app \
    && passwd -l root

RUN chown -R runner:runner /app
RUN chown -R runner:runner /home/runner

USER runner:runner

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

