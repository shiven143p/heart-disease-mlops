
"""Train and evaluate heart disease classification models."""

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.features.preprocess import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
)
from src.models.evaluate import (
    calculate_metrics,
    save_evaluation_plots,
    save_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_disease.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"

RANDOM_STATE = 42


def get_model_configurations():
    """Define candidate classifiers and hyperparameter grids."""

    return {
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            "parameters": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "parameters": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_split": [2, 5],
            },
        },
    }


def train_models():
    """Train candidates, compare CV results and evaluate the winner."""

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    configurations = get_model_configurations()

    results = []
    best_models = {}

    for model_name, configuration in configurations.items():

        print(f"\nTraining: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", configuration["estimator"]),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=configuration["parameters"],
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )

        search.fit(X_train, y_train)

        best_models[model_name] = search.best_estimator_

        result = {
            "model": model_name,
            "cv_roc_auc": float(search.best_score_),
            "best_parameters": search.best_params_,
        }

        results.append(result)

        print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")

    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values(
        "cv_roc_auc",
        ascending=False,
        kind="stable",
    )

    best_name = comparison.iloc[0]["model"]
    best_model = best_models[best_name]

    print(f"\nSelected model: {best_name}")

    test_metrics = calculate_metrics(
        best_model,
        X_test,
        y_test,
    )

    print("\nHeld-out test metrics:")

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the entire fitted preprocessing + classifier pipeline.
    model_path = MODEL_DIR / "heart_disease_pipeline.joblib"
    joblib.dump(best_model, model_path)

    # Save CV comparison.
    comparison.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
    )

    # Save final test metrics.
    save_metrics(
        test_metrics,
        REPORT_DIR / "test_metrics.csv",
    )

    # Save figures for report and later MLflow logging.
    save_evaluation_plots(
        best_model,
        X_test,
        y_test,
        REPORT_DIR,
    )

    # Save metadata for reproducibility.
    metadata = {
        "selected_model": best_name,
        "random_state": RANDOM_STATE,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "selection_metric": "cv_roc_auc",
        "best_parameters": {
            key: (
                value.item() if hasattr(value, "item") else value
            )
            for key, value in best_model.get_params().items()
            if key.startswith("classifier__")
        },
        "test_metrics": test_metrics,
    }

    with open(
        MODEL_DIR / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print(f"\nModel saved to: {model_path}")
    print(f"Evaluation artifacts saved to: {REPORT_DIR}")

    return best_model, comparison, test_metrics


if __name__ == "__main__":
    train_models()
