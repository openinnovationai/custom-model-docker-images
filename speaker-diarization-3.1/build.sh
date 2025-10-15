#!/bin/bash

echo "Building the docker image for ${OPERATOR}"

if [ -z "$OPERATOR" ]; then
    OPERATOR=cpu
fi

docker build -t pyannote-${OPERATOR} --build-arg OPERATOR=${OPERATOR} --platform linux/amd64 . 
