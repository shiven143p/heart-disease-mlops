
#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${K8S_NAMESPACE:-default}"
DEPLOYMENT="heart-disease-api"
CONTAINER="heart-disease-api"

echo "Starting Kubernetes rollback validation..."

# Read the currently deployed, known-good image.
GOOD_IMAGE=$(kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" \
  -o jsonpath='{.spec.template.spec.containers[0].image}')

echo "Known-good image: $GOOD_IMAGE"

# Confirm that the original deployment is healthy.
kubectl -n "$NAMESPACE" rollout status \
  deployment/"$DEPLOYMENT" --timeout=120s

# Deliberately use a nonexistent image.
BAD_IMAGE="invalid.example.invalid/heart-disease-api:broken"

echo "Attempting broken deployment: $BAD_IMAGE"

kubectl -n "$NAMESPACE" set image \
  deployment/"$DEPLOYMENT" \
  "$CONTAINER=$BAD_IMAGE"

# A successful rollout here would be unexpected.
if kubectl -n "$NAMESPACE" rollout status \
  deployment/"$DEPLOYMENT" --timeout=45s; then
    echo "ERROR: Broken deployment unexpectedly succeeded."
    exit 1
fi

echo "Broken deployment failed as expected."

echo "Rolling back to previous version..."

kubectl -n "$NAMESPACE" rollout undo \
  deployment/"$DEPLOYMENT"

kubectl -n "$NAMESPACE" rollout status \
  deployment/"$DEPLOYMENT" --timeout=180s

# Confirm the expected image was restored.
RESTORED_IMAGE=$(kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" \
  -o jsonpath='{.spec.template.spec.containers[0].image}')

if [[ "$RESTORED_IMAGE" != "$GOOD_IMAGE" ]]; then
  echo "ERROR: Rollback restored an unexpected image."
  echo "Expected: $GOOD_IMAGE"
  echo "Actual: $RESTORED_IMAGE"
  exit 1
fi

# Confirm all desired replicas are ready.
DESIRED=$(kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" \
  -o jsonpath='{.spec.replicas}')

READY=$(kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" \
  -o jsonpath='{.status.readyReplicas}')

if [[ "$READY" != "$DESIRED" ]]; then
  echo "ERROR: Only $READY of $DESIRED replicas are ready."
  exit 1
fi

echo "Rollback successful."
echo "Restored image: $RESTORED_IMAGE"
echo "Ready replicas: $READY/$DESIRED"
