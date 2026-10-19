"""Serve OpenAI Privacy Filter as a small fail-closed HTTP API."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field


PrivacyLabel = Literal[
    "private_person",
    "private_address",
    "private_email",
    "private_phone",
    "private_url",
    "private_date",
    "account_number",
    "secret",
]


class RedactRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=100_000)
    include_original: bool = False
    labels: list[PrivacyLabel] | None = Field(
        default=None,
        min_length=1,
        description="Categories to redact. Omit or use null for all categories.",
    )


def load_redactor() -> tuple[Any, str, str | None]:
    """Load and warm the model, returning it with device metadata."""
    import torch
    from opf import OPF

    requested = os.getenv("OPF_DEVICE")
    if requested not in {None, "cpu", "cuda"}:
        raise ValueError("OPF_DEVICE must be cpu or cuda")

    device = requested or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("OPF_DEVICE=cuda but PyTorch cannot access a CUDA device")

    device_name = torch.cuda.get_device_name(0) if device == "cuda" else None
    redactor = OPF(device=device)
    redactor.redact("Ready.")
    return redactor, device, device_name


def _apply_label_selection(result: dict[str, Any], labels: list[PrivacyLabel] | None) -> None:
    if labels is None:
        return

    selected = set(labels)
    spans = [span for span in result["detected_spans"] if span["label"] in selected]
    source = result["text"]
    pieces: list[str] = []
    cursor = 0
    for span in spans:
        pieces.extend([source[cursor : span["start"]], span["placeholder"]])
        cursor = span["end"]
    pieces.append(source[cursor:])

    result["redacted_text"] = "".join(pieces)
    result["detected_spans"] = spans
    result["summary"]["span_count"] = len(spans)
    result["summary"]["by_label"] = dict(sorted(Counter(span["label"] for span in spans).items()))


def _remove_originals(result: dict[str, Any]) -> None:
    result.pop("text", None)
    for span in result["detected_spans"]:
        span.pop("text", None)


def create_app(
    *,
    loader: Callable[[], tuple[Any, str, str | None]] = load_redactor,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        redactor, device, device_name = loader()
        application.state.redactor = redactor
        application.state.lock = Lock()
        application.state.device = device
        application.state.device_name = device_name
        yield
        del application.state.redactor

    application = FastAPI(title="OpenAI Privacy Filter", version="1.0.0", lifespan=lifespan)

    def health(request: Request) -> dict[str, str | None]:
        return {
            "status": "ok",
            "device": request.app.state.device,
            "device_name": request.app.state.device_name,
        }

    application.add_api_route("/health", health, methods=["GET"], include_in_schema=False)
    application.add_api_route("/health-check", health, methods=["GET"])
    application.add_api_route("/health", health, methods=["GET"])

    @application.post("/v1/redact")
    def redact(body: RedactRequest, request: Request) -> dict[str, Any]:
        # Bound GPU concurrency to one. Callers can retry a busy instance without
        # accumulating inference work or exhausting device memory.
        lock = request.app.state.lock
        if not lock.acquire(blocking=False):
            raise HTTPException(503, "Model is busy; retry shortly", headers={"Retry-After": "1"})

        try:
            result = request.app.state.redactor.redact(body.text).to_dict()
            _apply_label_selection(result, body.labels)
        except Exception:
            # Never return the unfiltered input as a successful fallback.
            raise HTTPException(500, "Privacy Filter inference failed") from None
        finally:
            lock.release()

        if not body.include_original:
            _remove_originals(result)
        return result

    return application


app = create_app()
