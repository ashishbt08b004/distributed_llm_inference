#!/usr/bin/env bash
# Phase 6 — cache-aware router tier (run ON the node).
#   config C (default): kv_router with prefill/decode orchestration
#   config B:          sglang_router --policy cache_aware (or kv_router, see ROUTER_IMPL_B)
source "$(dirname "$0")/stack.sh"
CFG="${1:-C}"

deploy_router "$CFG"
echo "$CFG" > "$BP1_ROOT/.active_config"
kubectl -n inference get pods -l tier=router

log "Two requests with the same long prefix should land on the same decode pod"
PREFIX=$(printf 'You are a meticulous assistant. %.0s' {1..40})
for i in 1 2; do
  kubectl -n inference exec deploy/bench-client -- curl -sf http://router.inference.svc.cluster.local:8080/v1/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_MODEL_NAME\",\"prompt\":\"$PREFIX Question $i: say hello.\",\"max_tokens\":8}" \
    | jq -c '{text: .choices[0].text}' || die "router request failed"
done
if kubectl -n inference get deploy kv-router >/dev/null 2>&1; then
  kubectl -n inference exec deploy/bench-client -- curl -s http://router.inference.svc.cluster.local:8080/metrics \
    | grep -E '^kv_router_(requests_total|prefix_matched)' || true
fi
pass "router answers on router.inference.svc:8080"
