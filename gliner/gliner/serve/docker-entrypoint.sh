#!/bin/bash
set -e

# Always resolve the bundled model from its image-owned cache. This remains stable
# when Kubernetes overrides HOME or injects the standard Hugging Face cache vars.
BUNDLED_HF_HOME="${GLINER_BUNDLED_HF_HOME:-/opt/gliner/huggingface}"
export HF_HOME="$BUNDLED_HF_HOME"
export HF_HUB_CACHE="$BUNDLED_HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$BUNDLED_HF_HOME/hub"
export TRANSFORMERS_CACHE="$BUNDLED_HF_HOME/hub"

CMD="python -m gliner.serve"
CMD="$CMD --port 8080"
CMD="$CMD --model ${GLINER_MODEL:-urchade/gliner_small-v2.1}"
CMD="$CMD --device ${GLINER_DEVICE:-cuda}"
CMD="$CMD --dtype ${GLINER_DTYPE:-float32}"
CMD="$CMD --max-batch-size ${GLINER_MAX_BATCH_SIZE:-32}"
CMD="$CMD --batch-wait-timeout-ms ${GLINER_BATCH_WAIT_MS:-5}"
CMD="$CMD --target-memory-fraction ${GLINER_MEMORY_FRACTION:-0.8}"

if [ "${GLINER_NUM_REPLICAS:-1}" != "1" ]; then
    CMD="$CMD --num-replicas ${GLINER_NUM_REPLICAS}"
fi

if [ "${GLINER_DISABLE_COMPILE}" = "true" ]; then
    CMD="$CMD --no-compile"
fi

if [ "${GLINER_ENABLE_FLASHDEBERTA}" = "true" ]; then
    CMD="$CMD --enable-flashdeberta"
fi

if [ "${GLINER_ENABLE_PACKING}" = "true" ]; then
    CMD="$CMD --enable-sequence-packing"
fi

if [ -n "${GLINER_QUANTIZATION}" ]; then
    CMD="$CMD --quantization ${GLINER_QUANTIZATION}"
fi

CMD="$CMD --route-prefix ${GLINER_ROUTE_PREFIX:-/}"

echo "Starting GLiNER Serve..."
echo "Command: $CMD"
exec $CMD
