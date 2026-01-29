# Image configuration
IMAGE_NAME ?= mxbai-rerank
TAG ?= latest
MODEL_PATH ?= $(shell pwd)/models

# Build settings
DOCKER_BUILDKIT = 1

.PHONY: help build-cpu build-nvidia run-cpu run-nvidia clean

help:
	@echo "Usage:"
	@echo "  make build-cpu      Build the CPU-only Docker image"
	@echo "  make build-nvidia   Build the NVIDIA GPU Docker image"
	@echo "  make run-cpu        Run the CPU-only container"
	@echo "  make run-nvidia     Run the NVIDIA GPU container"
	@echo "  make clean          Remove the Docker images"

build-cpu:
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build \
		--progress=plain \
		--build-arg OPERATOR=cpu \
		-t $(IMAGE_NAME):$(TAG)-cpu .

build-nvidia:
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build \
		--progress=plain \
		--build-arg OPERATOR=nvidia \
		-t $(IMAGE_NAME):$(TAG)-nvidia .

run-cpu:
	docker run -it --rm \
		-p 8080:8080 \
		-v $(MODEL_PATH):/app/model \
		-e OICM_MODEL_PATH=/app/model \
		$(IMAGE_NAME):$(TAG)-cpu

run-nvidia:
	docker run -it --rm \
		--gpus all \
		-p 8080:8080 \
		$(IMAGE_NAME):$(TAG)-nvidia

clean:
	docker rmi $(IMAGE_NAME):$(TAG)-cpu || true
	docker rmi $(IMAGE_NAME):$(TAG)-nvidia || true
