
"""Unit tests for saved-model inference utilities."""

import numpy as np
import pytest

from src.models.predict import (
    load_model,
    predict_heart_disease,
)


@pytest.fixture
def sample_patient():
    return {
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
    """Deterministic test double independent of trained artifacts."""

    classes_ = np.array([0, 1])

    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


def test_prediction_output(sample_patient):
    result = predict_heart_disease(
        FakeModel(),
        sample_patient,
    )

    assert result["prediction"] == 1
    assert result["risk"] == "heart_disease"
    assert result["probability"] == 0.8
    assert result["confidence"] == 0.8


def test_missing_feature_raises_error(sample_patient):
    patient = sample_patient.copy()
    patient.pop("age")

    with pytest.raises(ValueError, match="Missing required features"):
        predict_heart_disease(FakeModel(), patient)


def test_invalid_input_type():
    with pytest.raises(TypeError):
        predict_heart_disease(FakeModel(), [1, 2, 3])


def test_missing_model_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nonexistent.joblib")
