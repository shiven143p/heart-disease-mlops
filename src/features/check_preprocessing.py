
"""Smoke test preprocessing against the actual UCI dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.preprocess import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_disease.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor()

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print(f"Original dataset: {df.shape}")
    print(f"Training data: {X_train.shape}")
    print(f"Testing data: {X_test.shape}")

    print(
        "Transformed training data:",
        X_train_transformed.shape,
    )
    print(
        "Transformed testing data:",
        X_test_transformed.shape,
    )

    print(
        "Number of output features:",
        len(preprocessor.get_feature_names_out()),
    )

    assert X_train_transformed.shape[0] == len(X_train)
    assert X_test_transformed.shape[0] == len(X_test)
    assert np.isfinite(X_train_transformed).all()
    assert np.isfinite(X_test_transformed).all()

    print("\nPreprocessing smoke test passed.")


if __name__ == "__main__":
    main()
