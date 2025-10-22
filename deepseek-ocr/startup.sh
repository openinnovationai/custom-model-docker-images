#!/bin/bash

python3 -m vllm.entrypoints.openai.api_server \
    --host=127.0.0.1 \
    --port=8081 \
    --model=/home/runner/model \
    --served-model-name=deepseek-ai/DeepSeek-OCR &

VLLM_PID=$!

/usr/sbin/nginx -c /home/runner/nginx.conf

wait -n

EXIT_CODE=$?

kill $VLLM_PID

exit $EXIT_CODE