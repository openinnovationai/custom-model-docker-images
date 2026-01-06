"""Integration tests for API routes."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.server import app


@pytest.fixture
def client():
    """Create a test client with mocked ASR service."""
    with patch("app.service.asr_service") as mock_service:
        mock_service.load_model = MagicMock()
        with TestClient(app) as test_client:
            yield test_client


def test_get_health_check(client: TestClient):
    """Health check should return valid JSON."""
    response = client.get("/health-check")

    assert response.headers["content-type"] == "application/json"
    assert response.status_code == 200
    assert response.text == '"ok"'


def test_get_models(client: TestClient):
    """Models endpoint."""
    response = client.get("/v1/models")

    assert response.status_code == 200

    data = response.json()

    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) == 1

    model = response.json()["data"][0]

    assert "id" in model
    assert "object" in model
    assert "created" in model
    assert "owned_by" in model
    assert model["object"] == "model"
    assert model["owned_by"] == "omnilingual-asr"
    assert isinstance(model["created"], int)


@patch("app.server.MODEL_NAME", "custom_model_name")
def test_get_models_uses_configured_model_name(client: TestClient):
    """Model ID should reflect the configured MODEL_NAME."""
    response = client.get("/v1/models")
    model = response.json()["data"][0]

    assert model["id"] == "custom_model_name"
