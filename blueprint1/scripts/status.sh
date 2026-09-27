#!/usr/bin/env bash
# One-screen health check of every layer (run ON the node).
source "$(dirname "$0")/lib.sh"

echo "Active config: $(cat "$BP1_ROOT/.active_config" 2>/dev/null || echo none)"
for ns in gpu-operator hami-system observability kv-tier inference; do
  echo; log "namespace $ns"
  kubectl get pods -n "$ns" -o wide 2>/dev/null | awk 'NR==1 || $3!="Completed"' || echo "  (missing)"
done
echo; log "GPU (host view)"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
echo; log "HAMi allocations (node annotations)"
kubectl get node -o json | jq -r '.items[0].metadata.annotations | to_entries[] | select(.key|test("hami")) | "\(.key)=\(.value[0:160])"' 2>/dev/null || true
