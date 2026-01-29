.PHONY: build-cpu build-nvidia run-cpu run-nvidia clean
.DEFAULT_GOAL := help
build-cpu: # build cpu image
	docker build -t pyannote-cpu -f Dockerfile.cpu --platform linux/amd64 .

build-nvidia: # build nvidia image
	docker build -t pyannote-nvidia -f Dockerfile.gpu --platform linux/amd64 .

run-cpu: # Run cpu container
	docker run --rm -p 8080:8080 pyannote-cpu

run-nvidia: # Run nvidia container
	docker run --rm --gpus all -p 8080:8080 pyannote-nvidia

clean: # Clean temporary files
	@rm -rf $(TMP_PATH) __pycache__ .pytest_cache
	@find . -name '*.pyc' -delete
	@find . -name '__pycache__' -delete

help: # Show this help
	@egrep -h '\s#\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'