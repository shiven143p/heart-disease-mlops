
"""Train, tune, evaluate, and track heart disease models."""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
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

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_disease.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
)

RANDOM_STATE = 42

# SQLite tracking database
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

MLFLOW_TRACKING_URI = (
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
)


def configure_mlflow():
    """Configure MLflow with a SQLite tracking backend."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment(
        "heart-disease-classification"
    )


def get_model_configurations():
    """Return model configurations and tuning grids."""

    return {
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            "parameters": {
                "classifier__C": [
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                ],
                "classifier__class_weight": [
                    None,
                    "balanced",
                ],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "parameters": {
                "classifier__n_estimators": [
                    100,
                    200,
                ],
                "classifier__max_depth": [
                    None,
                    5,
                    10,
                ],
                "classifier__min_samples_split": [
                    2,
                    5,
                ],
            },
        },
    }


def train_models():
    """Train models and log experiments to MLflow."""

    configure_mlflow()

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Stratified train/test split
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    # Cross-validation configuration
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    configurations = get_model_configurations()

    results = []
    best_models = {}

    # Train both candidate models
    for model_name, configuration in configurations.items():

        print(f"\nTraining: {model_name}")

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "classifier",
                    configuration["estimator"],
                ),
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

        with mlflow.start_run(run_name=model_name) as run:

            # Log general parameters
            mlflow.log_params(
                {
                    "model_type": model_name,
                    "random_state": RANDOM_STATE,
                    "cv_folds": 5,
                    "test_size": 0.20,
                    "selection_metric": "roc_auc",
                }
            )

            # Train model
            search.fit(X_train, y_train)

            best_models[model_name] = (
                search.best_estimator_
            )

            # Log best hyperparameters
            mlflow.log_params(
                search.best_params_
            )

            # Log cross-validation metrics
            mlflow.log_metric(
                "best_cv_roc_auc",
                float(search.best_score_),
            )

            best_index = search.best_index_

            cv_std = search.cv_results_[
                "std_test_score"
            ][best_index]

            mlflow.log_metric(
                "cv_roc_auc_std",
                float(cv_std),
            )

            # Save CV results
            cv_results_path = (
                REPORT_DIR
                / f"{model_name}_cv_results.csv"
            )

            pd.DataFrame(
                search.cv_results_
            ).to_csv(
                cv_results_path,
                index=False,
            )

            mlflow.log_artifact(
                str(cv_results_path),
                artifact_path="cross_validation",
            )

            # Log complete fitted model pipeline
            mlflow.sklearn.log_model(
                sk_model=search.best_estimator_,
                name="model",
                serialization_format="cloudpickle",
            )

            # Record comparison information
            result = {
                "model": model_name,
                "cv_roc_auc": float(
                    search.best_score_
                ),
                "best_parameters": (
                    search.best_params_
                ),
                "run_id": run.info.run_id,
            }

            results.append(result)

            print(
                "Best CV ROC-AUC:",
                round(search.best_score_, 4),
            )

            print(
                "Best parameters:",
                search.best_params_,
            )

            print(
                "MLflow Run ID:",
                run.info.run_id,
            )

    # Compare models using CV scores only
    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        "cv_roc_auc",
        ascending=False,
        kind="stable",
    )

    best_name = comparison.iloc[0]["model"]
    best_model = best_models[best_name]

    print(f"\nSelected model: {best_name}")

    # Evaluate selected model on held-out test set
    test_metrics = calculate_metrics(
        best_model,
        X_test,
        y_test,
    )

    print("\nHeld-out test metrics:")

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save complete model pipeline
    model_path = (
        MODEL_DIR
        / "heart_disease_pipeline.joblib"
    )

    joblib.dump(
        best_model,
        model_path,
    )

    # Save model comparison
    comparison.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
    )

    # Save test metrics
    save_metrics(
        test_metrics,
        REPORT_DIR / "test_metrics.csv",
    )

    # Save evaluation plots
    save_evaluation_plots(
        best_model,
        X_test,
        y_test,
        REPORT_DIR,
    )

    # Log final test results only for selected model
    selected_run_id = next(
        result["run_id"]
        for result in results
        if result["model"] == best_name
    )

    with mlflow.start_run(
        run_id=selected_run_id
    ):

        mlflow.set_tag(
            "selected_model",
            "true",
        )

        mlflow.set_tag(
            "evaluation_status",
            "final_test",
        )

        # Log test metrics
        for metric_name, metric_value in (
            test_metrics.items()
        ):
            mlflow.log_metric(
                f"test_{metric_name}",
                float(metric_value),
            )

        # Log evaluation artifacts
        for filename in [
            "confusion_matrix.png",
            "roc_curve.png",
            "test_metrics.csv",
            "model_comparison.csv",
        ]:
            mlflow.log_artifact(
                str(REPORT_DIR / filename),
                artifact_path="evaluation",
            )

        # Log standalone model artifact
        mlflow.log_artifact(
            str(model_path),
            artifact_path="model_export",
        )

    # Save model metadata
    metadata = {
        "selected_model": best_name,
        "random_state": RANDOM_STATE,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "selection_metric": "cv_roc_auc",
        "best_parameters": next(
            result["best_parameters"]
            for result in results
            if result["model"] == best_name
        ),
        "test_metrics": test_metrics,
        "mlflow_run_id": selected_run_id,
    }

    with open(
        MODEL_DIR / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("\nTraining completed successfully.")
    print(f"Model saved to: {model_path}")
    print(f"MLflow database: {MLFLOW_DB_PATH}")
    print(f"Evaluation artifacts: {REPORT_DIR}")

    return best_model, comparison, test_metrics


if __name__ == "__main__":
    train_models()
