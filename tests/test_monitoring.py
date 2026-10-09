
"""Tests for monitoring and metrics endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


class FakeModel:
    classes_ = [0, 1]

    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        return [[0.2, 0.8]]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        "src.api.main.load_model",
        lambda: FakeModel(),
    )

    with TestClient(app) as test_client:
        yield test_client


def test_metrics_endpoint(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "heart_disease_http_requests_total" in response.text


def test_request_metrics_recorded(client):
    client.get("/health")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "heart_disease_http_request_duration_seconds" in response.text
