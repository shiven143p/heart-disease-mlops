# Virtual Lab Reproducibility Guide

## 1. Objective

This guide documents how to reproduce the Heart Disease MLOps project in the university's Virtual Lab using the same GitHub repository and published Docker image developed in GitHub Codespaces.

## 2. Verified Environment

| Component | Virtual Lab |
|---|---|
| Operating system | Rocky Linux 9.5 |
| Architecture | x86_64 |
| Python | 3.12.9 |
| CPU | 4 cores |
| Memory | 15 GiB |
| Docker | 28.0.1 |
| Docker Compose | 2.33.1 |
| Minikube | 1.35.0 |
| Kubernetes | v1.32.0 |
| Git | 2.43.5 |

## 3. Repository Setup

Clone the repository:

```bash
git clone https://github.com/<USER_NAME>/heart-disease-mlops.git
cd heart-disease-mlops
```

## 4. Docker Reproduction

Use the GHCR image reference from a successful GitHub Actions run.

```bash
docker pull ghcr.io/<USER_NAME>/heart-disease-mlops:179d3f01f9dcd2d38e8c57d60a2bb6581b2fde33

docker run -d \
  --name heart-disease-lab \
  -p 8000:8000 \
  ghcr.io/<USER_NAME>/heart-disease-mlops:179d3f01f9dcd2d38e8c57d60a2bb6581b2fde33
```

Verify health:

```bash
curl http://127.0.0.1:8000/health
```

Verify prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_patient.json
```

## 5. Kubernetes Reproduction

Start Minikube if no cluster is running:

```bash
minikube start --driver=docker --cpus=2 --memory=4096
```

Check node readiness:

```bash
minikube kubectl -- get nodes
```

Load the published image:

```bash
minikube image load \
  ghcr.io/<USER_NAME>/heart-disease-mlops:179d3f01f9dcd2d38e8c57d60a2bb6581b2fde33
```

Deploy using the reusable script:

```bash
bash scripts/deploy_kubernetes.sh \
  ghcr.io/<USER_NAME>/heart-disease-mlops:179d3f01f9dcd2d38e8c57d60a2bb6581b2fde33
```

The script automatically selects standalone kubectl or Minikube's bundled kubectl.

Verify the deployment:

```bash
minikube kubectl -- get deployments
minikube kubectl -- get pods
```

Start port forwarding:

```bash
minikube kubectl -- port-forward \
  service/heart-disease-service 8080:80
```

Use another terminal to test the Kubernetes API at `http://127.0.0.1:8080`.

## 6. Python Training Reproduction

Create an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest tests/ -v
```

Download and validate the dataset:

```bash
python -m src.data.download
python -m src.data.validate
```

Train models and log MLflow experiments:

```bash
python -m src.models.train
```

## 7. Reproduced Results

| Metric | Codespaces | Virtual Lab |
|---|---:|---:|
| Logistic Regression CV ROC-AUC | 0.9028 | 0.9028 |
| Random Forest CV ROC-AUC | 0.8984 | 0.8984 |
| Accuracy | 0.8852 | 0.8852 |
| Precision | 0.8387 | 0.8387 |
| Recall | 0.9286 | 0.9286 |
| F1-score | 0.8814 | 0.8814 |
| Test ROC-AUC | 0.9654 | 0.9654 |

All displayed evaluation metrics matched to four decimal places across the two environments.

## 8. Reproducibility Notes

The development environment used Python 3.14.2, while the Virtual Lab used Python 3.12.9.

The same training and evaluation code produced matching reported metrics, demonstrating reproducibility of the experiment under the tested configurations.

This does not guarantee byte-identical serialized model artifacts or compatibility with arbitrary future dependency versions.

## 9. Security and Limitations

- The classifier is an educational demonstration and is not clinically validated.
- Serialized models must only be loaded from trusted sources.
- Do not log identifiable patient data.
- Minikube is intended for development and demonstration rather than high-availability production hosting.
- The GHCR image must be accessible to the lab environment, either publicly or through authorized registry authentication.

## 10. Conclusion

The project was successfully reproduced across GitHub Codespaces and a Rocky Linux Virtual Lab, demonstrating portable container inference, Kubernetes deployment, and consistent machine learning training results.