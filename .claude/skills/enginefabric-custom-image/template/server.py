"""EngineFabric custom-image model server.

Contract: listen on 0.0.0.0:8080, GET /health-check returns 200 only once the
model is loaded, inference on a function-named route.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("model-server")


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[str] = Field(min_length=1, max_length=64)


class Label(BaseModel):
    label: str
    score: float


class ClassifyResponse(BaseModel):
    labels: list[Label]


def load_model() -> Any:
    """Load weights baked into the image. Never downloads."""
    import torch
    from transformers import pipeline

    requested = os.getenv("MODEL_DEVICE")  # "cuda", "cpu" or unset (auto)
    if requested not in {None, "", "cpu", "cuda"}:
        raise ValueError("MODEL_DEVICE must be cpu or cuda")
    device = requested or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        # Fail closed: a GPU deployment must not silently fall back to CPU.
        raise RuntimeError("MODEL_DEVICE=cuda but no CUDA device is visible")

    model_dir = os.environ["MODEL_DIR"]
    logger.info("loading model from %s on %s", model_dir, device)
    # A local directory + HF_HUB_OFFLINE=1 (set in the Dockerfile) means no hub calls.
    model = pipeline("text-classification", model=model_dir, device=device)
    model(["warm-up"])  # first call compiles kernels / allocates; keep it out of user latency
    logger.info("model ready")
    return model


def create_app(loader: Callable[[], Any] = load_model) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = None
        app.state.lock = Lock()
        app.state.model = loader()  # uvicorn accepts traffic only after this returns
        yield
        app.state.model = None

    app = FastAPI(title="model-server", lifespan=lifespan)

    @app.get("/health-check")
    def health_check(request: Request):
        if request.app.state.model is None:
            return JSONResponse(status_code=503, content={"status": "loading", "message": "Model is loading"})
        return {"status": "healthy", "message": "Model is ready and healthy"}

    @app.post("/v1/classify", response_model=ClassifyResponse)
    def classify(body: ClassifyRequest, request: Request) -> ClassifyResponse:
        # Sync def -> runs in the threadpool, so /health-check stays responsive
        # during inference. The lock serialises access to the single model copy.
        with request.app.state.lock:
            try:
                outputs = request.app.state.model(body.instances)
            except Exception:
                logger.exception("inference failed")
                raise HTTPException(500, "inference failed") from None
        return ClassifyResponse(labels=[Label(label=o["label"], score=float(o["score"])) for o in outputs])

    return app


app = create_app()
