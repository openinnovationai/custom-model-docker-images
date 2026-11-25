"""Contract tests with a fake model - no weights, no GPU, no network."""

from fastapi.testclient import TestClient

from server import create_app


def fake_model(texts):
    return [{"label": "POSITIVE", "score": 0.99} for _ in texts]


def client() -> TestClient:
    return TestClient(create_app(loader=lambda: fake_model))


def test_health_check_ok_after_load():
    with client() as c:
        r = c.get("/health-check")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"


def test_classify_returns_one_label_per_instance():
    with client() as c:
        r = c.post("/v1/classify", json={"instances": ["a", "b"]})
        assert r.status_code == 200
        assert [x["label"] for x in r.json()["labels"]] == ["POSITIVE", "POSITIVE"]


def test_rejects_unknown_fields_and_empty_input():
    with client() as c:
        assert c.post("/v1/classify", json={"instances": []}).status_code == 422
        assert c.post("/v1/classify", json={"instances": ["a"], "x": 1}).status_code == 422
