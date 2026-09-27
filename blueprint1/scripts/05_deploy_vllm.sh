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
if [ "$CFG" = "C" ]; then
  log "LMCache/Mooncake init lines from a decode pod:"
  kubectl -n inference logs vllm-decode-0 | grep -iE "lmcache|mooncake" | tail -8 || true
fi
pass "config $CFG vLLM pools are Ready and answer completions"
