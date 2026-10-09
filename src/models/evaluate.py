
"""Evaluation utilities for heart disease classifiers."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def calculate_metrics(model, X_test, y_test):
    """Calculate classification metrics on a held-out dataset."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def save_evaluation_plots(model, X_test, y_test, output_dir):
    """Save confusion matrix and ROC curve."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["Absent", "Present"],
        cmap="Blues",
    )

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )

    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close()


def save_metrics(metrics, output_path):
    """Persist metrics in a CSV file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(output_path, index=False)
