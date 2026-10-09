
"""Tests for the reusable preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.features.preprocess import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    build_preprocessor,
)


@pytest.fixture
def sample_data():
    """Small dataset containing representative feature values."""

    return pd.DataFrame(
        {
            "age": [45, 60, 52, 55],
            "trestbps": [120, 140, 130, 135],
            "chol": [220, 260, 240, 230],
            "thalach": [170, 140, 155, 150],
            "oldpeak": [0.0, 2.3, 1.2, 0.8],
            "sex": [1, 0, 1, 0],
            "cp": [1, 2, 3, 4],
            "fbs": [0, 1, 0, 0],
            "restecg": [0, 1, 2, 0],
            "exang": [0, 1, 0, 1],
            "slope": [1, 2, 3, 2],
            "ca": [0, 1, np.nan, 2],
            "thal": [3, 6, 7, np.nan],
        }
    )


def test_feature_count():
    """Ensure the expected 13 features are configured."""

    assert len(NUMERICAL_FEATURES) == 5
    assert len(CATEGORICAL_FEATURES) == 8
    assert len(FEATURE_COLUMNS) == 13
    assert len(set(FEATURE_COLUMNS)) == 13


def test_preprocessor_handles_missing_values(sample_data):
    """Missing values should be imputed successfully."""

    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(sample_data)

    assert transformed.shape[0] == len(sample_data)
    assert not np.isnan(transformed).any()


def test_preprocessor_produces_finite_values(sample_data):
    """All transformed values should be finite."""

    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(sample_data)

    assert np.isfinite(transformed).all()


def test_preprocessor_handles_unknown_categories(sample_data):
    """Unknown categories must not cause inference errors."""

    preprocessor = build_preprocessor()

    preprocessor.fit(sample_data)

    new_data = sample_data.iloc[[0]].copy()
    new_data["cp"] = 99

    transformed = preprocessor.transform(new_data)

    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()


def test_preprocessor_does_not_modify_original_data(sample_data):
    """Fitting and transforming should not mutate input data."""

    original = sample_data.copy(deep=True)

    preprocessor = build_preprocessor()
    preprocessor.fit_transform(sample_data)

    pd.testing.assert_frame_equal(sample_data, original)


def test_output_feature_names(sample_data):
    """Transformed features should have meaningful names."""

    preprocessor = build_preprocessor()
    preprocessor.fit(sample_data)

    names = preprocessor.get_feature_names_out()

    assert len(names) > len(NUMERICAL_FEATURES)
    assert any("numerical__age" == name for name in names)
    assert any("categorical__cp" in name for name in names)
