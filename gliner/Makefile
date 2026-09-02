.DEFAULT_GOAL := build
.PHONY: build build-nvidia export run run-nvidia

IMAGE ?= gliner-serve:amd64
GLINER_MODEL ?= urchade/gliner_small-v2.1
TAR_FILE ?= dist/gliner-serve-amd64.tar

# Load the single-platform image into the local Docker daemon.
build:
	docker buildx build --platform linux/amd64 --load \
		--build-arg GLINER_MODEL="$(GLINER_MODEL)" \
		-t "$(IMAGE)" -f gliner/serve/Containerfile .

build-nvidia: build

# Export the existing image, including model weights, without rebuilding or loading.
export:
	mkdir -p "$$(dirname "$(TAR_FILE)")"
	docker image save --output "$(TAR_FILE)" "$(IMAGE)"

run:
	docker run --rm --platform linux/amd64 --gpus all -p 8080:8080 "$(IMAGE)"

run-nvidia: run
