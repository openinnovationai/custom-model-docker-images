.PHONY: help lock test build build-gpu run run-gpu offline-smoke push
.DEFAULT_GOAL := help

IMAGE ?= my-model
TAG ?= cpu
REGISTRY ?=
GPU_BASE ?= nvidia/cuda:12.8.1-runtime-ubuntu24.04
NAME ?= $(IMAGE)-smoke

lock: # Resolve and pin dependencies into uv.lock
	uv lock

test: # Run contract tests with a fake model (no weights, no GPU)
	uv run --frozen pytest -q

build: # Build the linux/amd64 CPU image
	docker buildx build --platform linux/amd64 --load -t $(IMAGE):$(TAG) .

build-gpu: # Build the linux/amd64 CUDA image
	docker buildx build --platform linux/amd64 --load \
		--build-arg BASE_IMAGE=$(GPU_BASE) -t $(IMAGE):gpu .

run: # Run the CPU image on :8080
	docker run --rm --platform linux/amd64 -p 8080:8080 $(IMAGE):$(TAG)

run-gpu: # Run the GPU image on :8080
	docker run --rm --gpus all -p 8080:8080 $(IMAGE):gpu

offline-smoke: # Start with no network and check uid, health and predict from inside
	-docker rm -f $(NAME) >/dev/null 2>&1
	docker run -d --name $(NAME) --platform linux/amd64 --network none $(IMAGE):$(TAG)
	@for i in $$(seq 1 90); do \
		docker exec $(NAME) python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health-check',timeout=2)" 2>/dev/null && break; sleep 2; done
	docker exec $(NAME) id
	docker exec $(NAME) python -c "import urllib.request as u,json;print(u.urlopen('http://127.0.0.1:8080/health-check').read().decode());r=u.Request('http://127.0.0.1:8080/v1/classify',data=json.dumps({'instances':['great product']}).encode(),headers={'Content-Type':'application/json'});print(u.urlopen(r).read().decode())"
	docker rm -f $(NAME) >/dev/null

push: # Tag and push to $(REGISTRY)
	test -n "$(REGISTRY)"
	docker tag $(IMAGE):$(TAG) $(REGISTRY)/$(IMAGE):$(TAG)
	docker push $(REGISTRY)/$(IMAGE):$(TAG)

help: # Show this help
	@grep -E '^[a-z-]+:.*# ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*# "}; {printf "  %-14s %s\n", $$1, $$2}'
