.PHONY: build-nvidia build-cpu run-nvidia run-cpu
.DEFAULT_GOAL := help


build-nvidia: # Build nvidia image
	docker build --progress plain --build-arg OPERATOR=nvidia -t whisper-runner:nvidia .

build-cpu: # Build cpu image
	docker build --progress plain --build-arg OPERATOR=cpu -t whisper-runner:cpu .

run-nvidia: # Run nvidia image
	docker run --gpus all -p 8080:8080 whisper-runner:nvidia

run-cpu: # Run cpu image
	docker run -p 8080:8080 whisper-runner:cpu


help: # Show this help
	@egrep -h '\s#\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'