#!/bin/bash

echo "Building the docker image for ${OPERATOR}"

OPERATOR=${OPERATOR:-cpu}

if [ "$OPERATOR" = "cpu" ]; then
    DOCKERFILE="Dockerfile.cpu"
elif [ "$OPERATOR" = "nvidia" ]; then
    DOCKERFILE="Dockerfile.gpu"
else
    echo "Unknown OPERATOR: $OPERATOR"
    exit 1
fi

docker build \
  -t pyannote-${OPERATOR} \
  -f ${DOCKERFILE} \
  --progress=plain \
  --platform linux/amd64 .
