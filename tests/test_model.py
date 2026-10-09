
"""Unit tests for model construction and evaluation."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.preprocess import FEATURE_COLUMNS, build_preprocessor
from src.models.evaluate import calculate_metrics
from src.models.train import get_model_configurations


def test_two_model_configurations_exist():
    configs = get_model_configurations()

    assert "logistic_regression" in configs
    assert "random_forest" in configs


def test_model_hyperparameter_grids_are_defined():
    configs = get_model_configurations()

    for config in configs.values():
        assert len(config["parameters"]) > 0


def test_pipeline_can_train_and_predict():
    """Test a full preprocessing + classifier pipeline."""

    rng = np.random.default_rng(42)
    n = 30

    X = pd.DataFrame(
        {
            "age": rng.integers(30, 75, n),
            "trestbps": rng.integers(100, 180, n),
            "chol": rng.integers(150, 350, n),
            "thalach": rng.integers(90, 190, n),
            "oldpeak": rng.uniform(0, 4, n),
            "sex": rng.integers(0, 2, n),
            "cp": rng.integers(1, 5, n),
            "fbs": rng.integers(0, 2, n),
            "restecg": rng.integers(0, 3, n),
            "exang": rng.integers(0, 2, n),
            "slope": rng.integers(1, 4, n),
            "ca": rng.integers(0, 4, n),
            "thal": rng.choice([3, 6, 7], n),
        }
    )

    X = X[FEATURE_COLUMNS]
    y = np.array([0, 1] * 15)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(max_iter=1000),
            ),
        ]
    )

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    assert len(predictions) == n
    assert probabilities.shape == (n, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    metrics = calculate_metrics(pipeline, X, y)

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    assert all(0 <= value <= 1 for value in metrics.values())
