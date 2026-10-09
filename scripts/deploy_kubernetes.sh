
#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:?Usage: bash scripts/deploy_kubernetes.sh IMAGE}"
NAMESPACE="${K8S_NAMESPACE:-default}"
DEPLOYMENT="heart-disease-api"

# Select a Kubernetes command available in this environment.
if command -v kubectl >/dev/null 2>&1; then
    KUBECTL=(kubectl)
elif command -v minikube >/dev/null 2>&1; then
    KUBECTL=(minikube kubectl --)
else
    echo "ERROR: Neither kubectl nor minikube is available."
    exit 1
fi

echo "Deploying image: $IMAGE"
echo "Namespace: $NAMESPACE"

# Render a version-specific Deployment manifest.
python3 - "$IMAGE" <<'PY'
import sys
import tempfile
from pathlib import Path

image = sys.argv[1]

source = Path("kubernetes/deployment-registry.yaml")
template = source.read_text(encoding="utf-8")

placeholder = "ghcr.io/example/placeholder:latest"

if template.count(placeholder) != 1:
    raise RuntimeError("Expected exactly one image placeholder")

rendered = template.replace(placeholder, image)

output = Path(tempfile.gettempdir()) / "heart-disease-deployment.yaml"
output.write_text(rendered, encoding="utf-8")

print(f"Rendered manifest: {output}")
PY

# Apply the Kubernetes Service.
"${KUBECTL[@]}" -n "$NAMESPACE" apply \
    -f kubernetes/service.yaml

# Apply the generated Deployment.
"${KUBECTL[@]}" -n "$NAMESPACE" apply \
    -f "${TMPDIR:-/tmp}/heart-disease-deployment.yaml"

# Wait for Kubernetes to finish updating the pods.
"${KUBECTL[@]}" -n "$NAMESPACE" rollout status \
    deployment/"$DEPLOYMENT" \
    --timeout=180s

# Display deployment evidence.
"${KUBECTL[@]}" -n "$NAMESPACE" get deployments
"${KUBECTL[@]}" -n "$NAMESPACE" get pods \
    -l app=heart-disease-api
"${KUBECTL[@]}" -n "$NAMESPACE" get services

echo "Deployment successful: $IMAGE"
