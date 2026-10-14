from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient

from server import create_app


@dataclass
class FakeResult:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.payload


class FakeRedactor:
    def redact(self, text: str) -> FakeResult:
        if text == "sensitive-value":
            raise RuntimeError(f"must not leak: {text}")
        return FakeResult(
            {
                "schema_version": 1,
                "text": text,
                "redacted_text": "Email <PRIVATE_PERSON> at <PRIVATE_EMAIL>.",
                "detected_spans": [
                    {
                        "label": "private_person",
                        "start": 6,
                        "end": 11,
                        "text": "Alice",
                        "placeholder": "<PRIVATE_PERSON>",
                    },
                    {
                        "label": "private_email",
                        "start": 15,
                        "end": 32,
                        "text": "alice@example.com",
                        "placeholder": "<PRIVATE_EMAIL>",
                    },
                ],
                "summary": {
                    "span_count": 2,
                    "by_label": {"private_email": 1, "private_person": 1},
                },
            }
        )


def loader():
    return FakeRedactor(), "cuda", "NVIDIA B300"


def test_health_check_reports_loaded_device():
    with TestClient(create_app(loader=loader)) as client:
        response = client.get("/health-check")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "device": "cuda", "device_name": "NVIDIA B300"}


def test_redact_returns_full_redacted_text_without_original_values():
    with TestClient(create_app(loader=loader)) as client:
        response = client.post("/v1/redact", json={"text": "Email Alice at alice@example.com."})

    assert response.status_code == 200
    payload = response.json()
    assert payload["redacted_text"] == "Email <PRIVATE_PERSON> at <PRIVATE_EMAIL>."
    assert "text" not in payload
    assert all("text" not in span for span in payload["detected_spans"])


def test_request_can_select_redaction_categories():
    with TestClient(create_app(loader=loader)) as client:
        response = client.post(
            "/v1/redact",
            json={
                "text": "Email Alice at alice@example.com.",
                "labels": ["private_email"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["redacted_text"] == "Email Alice at <PRIVATE_EMAIL>."
    assert payload["summary"] == {"span_count": 1, "by_label": {"private_email": 1}}


def test_inference_failure_is_fail_closed():
    with TestClient(create_app(loader=loader)) as client:
        response = client.post("/v1/redact", json={"text": "sensitive-value"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Privacy Filter inference failed"}
    assert "sensitive-value" not in response.text


def test_busy_instance_returns_retryable_response():
    with TestClient(create_app(loader=loader)) as client:
        client.app.state.lock.acquire()
        try:
            response = client.post("/v1/redact", json={"text": "Email Alice at alice@example.com."})
        finally:
            client.app.state.lock.release()

    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
