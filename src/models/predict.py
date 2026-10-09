
"""Reusable inference utilities for heart disease prediction."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.preprocess import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "heart_disease_pipeline.joblib"
)


def load_model(model_path=None):
    """Load a trusted, previously trained ML pipeline."""

    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. "
            "Run 'python -m src.models.train' first."
        )

    model = joblib.load(path)

    if not hasattr(model, "predict"):
        raise TypeError("Loaded artifact does not support prediction.")

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Loaded artifact does not support probability prediction."
        )

    return model


def predict_heart_disease(model, patient_data):
    """Predict heart disease presence from patient feature values.

    Parameters
    ----------
    model:
        Fitted preprocessing + classifier pipeline.
    patient_data:
        Dictionary containing all 13 input features.

    Returns
    -------
    dict:
        Prediction, positive-class probability, and confidence.
    """

    if not isinstance(patient_data, dict):
        raise TypeError("Patient data must be a dictionary.")

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in patient_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    input_df = pd.DataFrame(
        [{feature: patient_data[feature] for feature in FEATURE_COLUMNS}]
    )

    prediction = int(model.predict(input_df)[0])

    probabilities = model.predict_proba(input_df)[0]

    classes = list(model.classes_)
    positive_index = classes.index(1)

    positive_probability = float(probabilities[positive_index])

    confidence = float(np.max(probabilities))

    return {
        "prediction": prediction,
        "risk": (
            "heart_disease"
            if prediction == 1
            else "no_heart_disease"
        ),
        "probability": round(positive_probability, 4),
        "confidence": round(confidence, 4),
    }
