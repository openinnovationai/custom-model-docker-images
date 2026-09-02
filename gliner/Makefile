.DEFAULT_GOAL := build
.PHONY: build build-nvidia run run-nvidia

IMAGE ?= gliner-serve:amd64
GLINER_MODEL ?= urchade/gliner_small-v2.1

# Load the single-platform image into the local Docker daemon.
build:
	docker buildx build --platform linux/amd64 --load \
		--build-arg GLINER_MODEL="$(GLINER_MODEL)" \
		-t "$(IMAGE)" -f gliner/serve/Containerfile .

build-nvidia: build

run:
	docker run --rm --platform linux/amd64 --gpus all -p 8080:8080 "$(IMAGE)"

run-nvidia: run
