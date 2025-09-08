FROM cupy/cupy:v13.2.0 

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user `runner` with UID/GID 10000
RUN groupadd -g 10000 runner && \
    useradd -m -u 10000 -g 10000 -s /bin/bash runner

COPY . .

RUN chown -R runner:runner /app

EXPOSE 8080

USER runner:runner

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

