FROM public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:6f890c0b3499688fff07993aa095cf90f4b00176

USER root

RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*

RUN useradd --uid 10000 runner
RUN mkdir /home/runner 

RUN cp /etc/nginx/mime.types /home/runner/mime.types

RUN mkdir -p /home/runner/logs 

RUN chown -R runner:runner /home/runner

USER runner
WORKDIR /home/runner

COPY ./model /home/runner/model

COPY --chown=runner:runner nginx.conf /home/runner/nginx.conf

ENV HF_HOME=/home/runner/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/runner/.cache/huggingface/transformers \
    TORCH_HOME=/home/runner/.cache/torch \
    PIP_CACHE_DIR=/home/runner/.cache/pip \
    XDG_CACHE_HOME=/home/runner/.cache

EXPOSE 8080

COPY --chown=runner:runner startup.sh /home/runner/startup.sh

RUN chmod +x /home/runner/startup.sh

ENTRYPOINT ["/home/runner/startup.sh"]