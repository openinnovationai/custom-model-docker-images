#!/bin/bash
#
# Build script for Omnilingual-ASR server Docker image.
#
# This script builds (and optionally pushes) the Docker image for the
# Omnilingual-ASR server.
#
# Environment Variables:
#
#   MODEL_NAME    - Name of the model to build (default: omniASR_LLM_300M_v2)
#                   The model name is used to generate the image tag suffix.
#                   Example: omniASR_LLM_1B_v2
#
#   NAMESPACE     - Namespace/registry prefix for the image name (optional)
#                   If provided, images will be tagged as NAMESPACE/omniasr-server
#                   Example: abc/omniasr-server
#                   If not provided, defaults to omniasr-server
#
#   LATEST_TAG    - Set to "true" to also tag the image as "latest"
#                   (default: false)
#
#   PUSH          - Set to "true" to push the image to the registry after building
#                   (default: false)
#
# Example usage:
#
#   # Build with default model name
#   bash build.sh
#
#   # Build with another variant model name
#   MODEL_NAME=omniASR_LLM_1B_v2 bash build.sh
#
#   # Build and tag as latest
#   LATEST_TAG=true bash build.sh
#
#   # Build and push to registry
#   PUSH=true bash build.sh
#
#   # Build with another variant model, tag as latest, and push
#   MODEL_NAME=omniASR_LLM_1B_v2 LATEST_TAG=true PUSH=true bash build.sh
#
#   # Build with namespace
#   NAMESPACE=abc bash build.sh
#
#   # Build with namespace and push
#   NAMESPACE=abc PUSH=true bash build.sh
#


MODEL_NAME=${MODEL_NAME:-omniASR_LLM_300M_v2}
BASE_TAG=cu126-pt280

# Convert model name to tag suffix, e.g.:
#     omniASR_LLM_300M_v2 -> llm-300m-v2
#     omniASR_CTC_300M_v2 -> ctc-300m-v2
#     omniASR_LLM_Unlimited_300M_v2 -> llm-unlimited-300m-v2
TAG_SUFFIX=$(echo $MODEL_NAME | sed 's/^omniASR_//' | tr 'A-Z_' 'a-z-')

# Build image name with optional namespace
if [ -n "$NAMESPACE" ]; then
    IMAGE_NAME="$NAMESPACE/omniasr-server"
else
    IMAGE_NAME="omniasr-server"
fi

# Build tags
TAGS="-t $IMAGE_NAME:$BASE_TAG-$TAG_SUFFIX"

# Handle latest tag
if [ "${LATEST_TAG:-false}" = "true" ]; then
    TAGS="$TAGS -t $IMAGE_NAME:latest"
fi

# Build command
BUILD_CMD="docker buildx build \
    --platform linux/amd64 \
    --build-arg MODEL_NAME=$MODEL_NAME \
    $TAGS"

# Optionally push
if [ "${PUSH:-false}" = "true" ]; then
    BUILD_CMD="$BUILD_CMD --push"
fi

# Execute build command
$BUILD_CMD .

echo "Docker image built successfully! You can run it with:"
echo "    docker run --gpus all -p 8080:8080 $IMAGE_NAME:$BASE_TAG-$TAG_SUFFIX"
