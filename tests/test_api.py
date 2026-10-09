
"""Unit tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

SAMPLE_PATIENT = {
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.2,
    "slope": 2,
    "ca": 0,
    "thal": 3,
}


class FakeModel:
    """A simple fake model for API testing."""

    classes_ = [0, 1]

    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        return [[0.2, 0.8]]


@pytest.fixture
def client(monkeypatch):
    """Create an API client without loading the real model."""

    monkeypatch.setattr(
        "src.api.main.load_model",
        lambda: FakeModel(),
    )

    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    """Verify the health endpoint."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction(client):
    """Verify successful prediction."""

    response = client.post(
        "/predict",
        json=SAMPLE_PATIENT,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == 1
    assert result["risk"] == "heart_disease"
    assert result["probability"] == 0.8
    assert result["confidence"] == 0.8


def test_missing_feature(client):
    """Reject requests missing a required feature."""

    patient = SAMPLE_PATIENT.copy()
    patient.pop("age")

    response = client.post(
        "/predict",
        json=patient,
    )

    assert response.status_code == 422


def test_invalid_category(client):
    """Reject unsupported thal categories."""

    patient = SAMPLE_PATIENT.copy()
    patient["thal"] = 5

    response = client.post(
        "/predict",
        json=patient,
    )

    assert response.status_code == 422
