# Reproducibility Guide

## Project Overview

This project implements an end-to-end machine learning workflow for binary heart disease classification using the UCI Heart Disease dataset.

The trained artifact contains the complete preprocessing and classification pipeline.

## Requirements

- Python 3.14.2 for reproducing the initial development environment
- Git
- Internet connectivity for the initial UCI dataset download
- Compatible operating system and Python dependencies

## Clean Setup

Clone the repository:

```bash
git clone <repository-url>
cd heart-disease-mlops
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate using:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements-lock.txt
```

## Reproduce Dataset

```bash
python -m src.data.download
python -m src.data.validate
```

## Run Unit Tests

```bash
python -m pytest tests/ -v
```

## Train and Track Models

```bash
python -m src.models.train
```

The training script performs hyperparameter tuning, cross-validation, model selection, and final evaluation.

Experiment metadata is stored in a local SQLite database.

## Open MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open the forwarded port or http://127.0.0.1:5000.

## Verify Saved Model

```bash
python -m src.models.check_inference
```

## Verify Environment

```bash
python -m src.check_environment
```

## Expected Artifacts

- `models/heart_disease_pipeline.joblib`
- `models/model_metadata.json`
- `reports/model_evaluation/`
- `mlflow.db`
- MLflow model artifacts

## Reproducibility Notes

- The dataset split uses random seed 42.
- Stratified cross-validation uses five folds.
- Preprocessing is fitted only on training data.
- The model artifact includes both preprocessing and classifier.
- Dependency versions are recorded in `requirements-lock.txt`.
- Model artifacts should only be loaded from trusted sources.
- The classifier is for educational demonstration and is not clinically validated.