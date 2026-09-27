#!/usr/bin/env bash
# Phase 4 — build the app image and deploy the Mooncake KV tier (run ON the node).
#
# The blueprint's `mooncakelabs/mooncake:v0.3.5` image does not exist. Mooncake
# ships as a pip wheel (mooncake-transfer-engine), so we bake it into one image
# with vLLM + LMCache and import that straight into k3s's containerd.
source "$(dirname "$0")/lib.sh"
need docker; need kubectl

if [ "${SKIP_IMAGE_BUILD:-false}" != "true" ]; then
  log "Building $APP_IMAGE from $BASE_IMAGE (first build pulls ~20 GB, 10-15 min)"
  sudo docker build -t "$APP_IMAGE" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg MOONCAKE_PIP_VERSION="$MOONCAKE_PIP_VERSION" \
    "$BP1_ROOT/image"
  log "Importing $APP_IMAGE into k3s containerd"
  sudo docker save "$APP_IMAGE" | sudo k3s ctr images import -
fi
sudo k3s ctr images ls -q | grep -q "${APP_IMAGE}" || die "$APP_IMAGE is not in k3s containerd"

log "GPU check: import vLLM / LMCache / Mooncake on a 2 GB slice"
kubectl delete pod image-gpu-check --ignore-not-found >/dev/null
apply_tpl "$BP1_ROOT/manifests/cluster/image-gpu-check.yaml"
wait_until "image-gpu-check to finish" 600 bash -c \
  "kubectl get pod image-gpu-check -o jsonpath='{.status.phase}' | grep -qE 'Succeeded|Failed'"
OUT=$(kubectl logs image-gpu-check 2>&1 || true)
kubectl delete pod image-gpu-check >/dev/null
echo "$OUT" | grep -v '^\[HAMI-core' | tail -15
echo "$OUT" | grep -q "GPU CHECK OK" || die "the image does not load on the GPU (see the error above). Fix it before Phase 5."
pass "image loads vLLM on the GPU"

ensure_ns kv-tier
mkdir -p "$MOONCAKE_DIR"
kubectl -n kv-tier create configmap mooncake-store-node \
  --from-file="$BP1_ROOT/files/mooncake_store_node.py" --dry-run=client -o yaml | kubectl apply -f -
apply_tpl "$BP1_ROOT/manifests/kv-tier/mooncake.yaml"

wait_rollout kv-tier deploy/mooncake-master 300
wait_rollout kv-tier deploy/mooncake-store 300

kubectl get pods -n kv-tier
log "Master log tail:"
kubectl logs -n kv-tier deploy/mooncake-master --tail=15
log "Store log tail:"
kubectl logs -n kv-tier deploy/mooncake-store --tail=5
kubectl logs -n kv-tier deploy/mooncake-store | grep -q "store node ready" \
  || die "store node did not register with the master — see logs above"
pass "Mooncake master + store Running; store contributed ${MOONCAKE_STORE_GB} GB to the pool"
