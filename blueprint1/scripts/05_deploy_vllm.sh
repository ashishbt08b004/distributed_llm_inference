#!/usr/bin/env bash
# Phase 5 — vLLM prefill + decode pools (full config C), run ON the node.
# Pass a config letter to deploy a different pool layout: ./05_deploy_vllm.sh B
source "$(dirname "$0")/stack.sh"
CFG="${1:-C}"

ensure_ns inference
kubectl -n inference create secret generic hf-token --from-literal=token="${HF_TOKEN}" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n inference create configmap vllm-entrypoint \
  --from-file="$BP1_ROOT/files/vllm-entrypoint.sh" --dry-run=client -o yaml | kubectl apply -f -

if ! kubectl -n inference get job model-download -o jsonpath='{.status.succeeded}' 2>/dev/null | grep -q 1; then
  log "Pre-downloading $MODEL_ID into $HF_CACHE_DIR"
  kubectl -n inference delete job model-download --ignore-not-found >/dev/null
  apply_tpl "$BP1_ROOT/manifests/inference/model-download-job.yaml"
  kubectl -n inference wait --for=condition=complete job/model-download --timeout=1800s
fi

deploy_pools "$CFG"
kubectl -n inference get pods -l app=vllm -o wide

# --- Pass gate: hit one pod of each pool directly -------------------------------------
gate() {
  local url="$1"
  kubectl -n inference exec deploy/bench-client -- true >/dev/null 2>&1 || {
    apply_tpl "$BP1_ROOT/manifests/bench/bench-client.yaml" >/dev/null
    kubectl -n inference create configmap bench-scripts --from-file="$BP1_ROOT/bench" \
      --dry-run=client -o yaml | kubectl apply -f - >/dev/null
    wait_rollout inference deploy/bench-client 300 >/dev/null
  }
  kubectl -n inference exec deploy/bench-client -- curl -sf "$url/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_MODEL_NAME\",\"prompt\":\"The capital of France is\",\"max_tokens\":10,\"temperature\":0}"
}
for url in $(metric_targets "$CFG"); do
  [[ "$url" == *":8000" ]] || continue
  log "Completion from $url"
  OUT=$(gate "$url") || die "request to $url failed"
  echo "$OUT" | jq -r '.choices[0].text'
  echo "$OUT" | grep -qi paris || warn "no 'Paris' in the answer from $url"
done
if [ "$CFG" = "C" ] && [ "${ENABLE_KV_TIER_C:-true}" = "true" ]; then
  log "LMCache/Mooncake init lines from a decode pod:"
  kubectl -n inference logs vllm-decode-0 | grep -iE "lmcache|mooncake" | tail -8 || true

  # A pod can report Ready with a broken Mooncake client (LMCache logs the failure
  # and carries on), then segfault on its first KV load. Catch both here.
  for pod in $(kubectl -n inference get pods -l app=vllm -o name); do
    if kubectl -n inference logs "$pod" | grep -qE "Client not available|Failed to create client|setup failed"; then
      die "$pod: Mooncake client did not initialise (see: kubectl -n inference logs ${pod#pod/} | grep -iE 'mooncake|client')"
    fi
  done
  log "Exercising the KV tier: a ~1.5k-token prompt twice (above LMCache's ${LMCACHE_CHUNK_SIZE}-token chunk)"
  RESTARTS_BEFORE=$(kubectl -n inference get pods -l app=vllm -o jsonpath='{range .items[*]}{.status.containerStatuses[0].restartCount}{" "}{end}')
  LONG=$(printf 'The quarterly report covers revenue, margins, regional growth and support tickets. %.0s' {1..90})
  for i in 1 2; do
    gate_url="$(pod_urls vllm-decode 1)"
    kubectl -n inference exec deploy/bench-client -- curl -sf "$gate_url/v1/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$SERVED_MODEL_NAME\",\"prompt\":\"$LONG Summarise in one word:\",\"max_tokens\":4}" >/dev/null \
      || die "long-prompt request $i failed: the KV tier is not working (check vllm-decode-0 logs)"
  done
  sleep 5
  RESTARTS_AFTER=$(kubectl -n inference get pods -l app=vllm -o jsonpath='{range .items[*]}{.status.containerStatuses[0].restartCount}{" "}{end}')
  [ "$RESTARTS_BEFORE" = "$RESTARTS_AFTER" ] || die "a vLLM pod restarted during the KV-tier check (restarts $RESTARTS_BEFORE -> $RESTARTS_AFTER)"
  kubectl -n inference logs vllm-decode-0 --since=2m | grep -iE "LMCache hit tokens|Stored|retriev" | tail -3 || true
fi
pass "config $CFG vLLM pools are Ready and answer completions"
