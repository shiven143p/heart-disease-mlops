
#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:?Usage: deploy_kubernetes.sh IMAGE}"
NAMESPACE="${K8S_NAMESPACE:-default}"
DEPLOYMENT="heart-disease-api"

echo "Deploying: $IMAGE"

# Render the manifest with the exact image version.
python - "$IMAGE" <<'PY'
import sys
from pathlib import Path

image = sys.argv[1]
source = Path("kubernetes/deployment-registry.yaml")
template = source.read_text(encoding="utf-8")

placeholder = "ghcr.io/example/placeholder:latest"
assert template.count(placeholder) == 1

rendered = template.replace(placeholder, image)
Path("/tmp/heart-disease-deployment.yaml").write_text(
    rendered, encoding="utf-8"
)
PY

# Apply the Service and rendered Deployment.
kubectl -n "$NAMESPACE" apply \
  -f kubernetes/service.yaml

kubectl -n "$NAMESPACE" apply \
  -f /tmp/heart-disease-deployment.yaml

# Verify the rollout.
kubectl -n "$NAMESPACE" rollout status \
  deployment/"$DEPLOYMENT" \
  --timeout=180s

kubectl -n "$NAMESPACE" get deployments
kubectl -n "$NAMESPACE" get pods \
  -l app=heart-disease-api
kubectl -n "$NAMESPACE" get services

echo "Deployment successful: $IMAGE"
