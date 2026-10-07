# Heart Disease Prediction - MLOps Project

End-to-end Machine Learning Operations project developed as part of the
M.Tech AI/ML MLOps assignment.

## Problem Statement

Build a machine learning classifier to predict the risk of heart disease
from patient health data and deploy the solution as a cloud-ready,
monitored API.

## Dataset

UCI Heart Disease Dataset.

## Project Objectives

The project demonstrates:

- Data acquisition and exploratory data analysis
- Data preprocessing and feature engineering
- Machine learning model development
- Cross-validation and hyperparameter tuning
- Experiment tracking using MLflow
- Model packaging and reproducibility
- Automated testing
- CI/CD using GitHub Actions
- REST API using FastAPI
- Docker containerization
- Kubernetes deployment
- Monitoring using Prometheus and Grafana

## Technology Stack

- Python
- Pandas / NumPy
- scikit-learn
- MLflow
- FastAPI
- pytest
- GitHub Actions
- Docker
- Kubernetes / Minikube
- Prometheus
- Grafana

Preinstallation of Libararies:
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
ucimlrepo
pytest
mlflow
fastapi
uvicorn
pydantic
prometheus-client
httpx
ruff

Commands to execute in sequence:
    python --version
    python -m venv .venv
    cmd: .\.venv\Scripts\Activate.ps1 / bash: source venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

    Validate Installed Libraries:
    python -c "import pandas, sklearn, matplotlib, seaborn; print('Environment OK')"

    For Testing:
    pytest --version
    jupyter --version