#!/usr/bin/env bash
# Phase 2 — HAMi fractional GPU sharing (run ON the node).
source "$(dirname "$0")/lib.sh"
need kubectl; need helm

NODE=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
kubectl label node "$NODE" gpu=on --overwrite

# HAMi's scheduler extender runs a kube-scheduler image that must match the
# cluster version (the blueprint hard-coded v1.29.0; k3s is newer).
K8S_VERSION=$(kubectl version -o json | jq -r .serverVersion.gitVersion | sed 's/+.*//')
log "Cluster version $K8S_VERSION"

helm repo add hami-charts https://project-hami.github.io/HAMi/ >/dev/null 2>&1 || true
helm repo update >/dev/null
ensure_ns hami-system
log "Installing HAMi"
helm upgrade --install hami hami-charts/hami -n hami-system \
  ${HAMI_CHART_VERSION:+--version "$HAMI_CHART_VERSION"} \
  --set scheduler.kubeScheduler.imageTag="$K8S_VERSION" \
  --wait --timeout 10m

kubectl get pods -n hami-system
wait_until "node to advertise nvidia.com/gpu" 180 bash -c \
  "kubectl get node $NODE -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' | grep -q '[1-9]'"

log "Node GPU capacity:"
kubectl get node "$NODE" -o json | jq '.status.capacity | with_entries(select(.key|test("nvidia|hami")))'
# HAMi registers memory/cores through its scheduler (node annotations), so they
# don't always show up in capacity; the canary below is the real test.

log "Canary: 8 GB / 20% slice"
kubectl delete pod hami-canary --ignore-not-found >/dev/null
render "$BP1_ROOT/manifests/cluster/hami-canary.yaml" | kubectl apply -f -
wait_until "hami-canary to finish" 300 bash -c \
  "kubectl get pod hami-canary -o jsonpath='{.status.phase}' | grep -qE 'Succeeded|Failed'"
OUT=$(kubectl logs hami-canary)
kubectl delete pod hami-canary >/dev/null
echo "$OUT"
# The GPU row looks like "NVIDIA H100 PCIe, 0, 8192" (noheader,nounits). HAMi's
# shim also logs "[HAMI-core Msg ...]" lines around it, so match the row itself
# instead of taking the last line.
TOTAL_MIB=$(echo "$OUT" | grep -E '^[^[].*, *[0-9]+, *[0-9]+ *$' | head -1 | awk -F', *' '{print $3}' | tr -dc 0-9)
if [ -n "$TOTAL_MIB" ] && (( TOTAL_MIB > 0 && TOTAL_MIB <= 8192 )); then
  pass "canary sees ${TOTAL_MIB} MiB — HAMi is slicing the H100"
else
  die "canary saw '${TOTAL_MIB:-?}' MiB (expected <= 8192). Check the pod used schedulerName: hami-scheduler and that the default runtime is nvidia."
fi
