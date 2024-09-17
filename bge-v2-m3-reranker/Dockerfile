FROM registry.gitlab.com/openinnovationai/platform/mlops/mlops-serving/rocm-6.1-pytorch-2.1.2-flash-attn

USER root

ARG DEBIAN_FRONTEND=noninteractive
EXPOSE 8080 

RUN apt-get update && apt-get install -y \
    wget git curl ffmpeg \
    nginx supervisor \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

RUN pip uninstall -y torch torchvision

RUN python3 -m pip install --no-cache-dir --pre torch==2.5.0.dev20240726 torchvision==0.20.0.dev20240726 --index-url https://download.pytorch.org/whl/nightly/rocm6.1

COPY proxy.conf /etc/nginx/conf.d/proxy.conf

RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

COPY app/app.py app.py
COPY app/model.py model.py
COPY deployment-config.yaml /app/deployment-config.yaml

COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

ENV SERVER_START_CMD /usr/bin/supervisord
ENTRYPOINT ["/usr/bin/supervisord"]