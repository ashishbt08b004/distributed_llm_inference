#!/usr/bin/env bash
# Phase 7 — LiteLLM gateway (run ON the node).
source "$(dirname "$0")/lib.sh"
ensure_gateway_key

apply_tpl "$BP1_ROOT/manifests/gateway/litellm.yaml"
kubectl -n inference rollout restart deploy/litellm >/dev/null   # pick up config changes
wait_rollout inference deploy/litellm 600

log "End-to-end through the gateway (NodePort 30400)"
curl -sf http://localhost:30400/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"1+1=? Answer with one number.\"}],\"max_tokens\":5}" \
  | jq -r '.choices[0].message.content' || die "gateway chat request failed"

CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:30400/v1/chat/completions \
  -H 'Authorization: Bearer wrong-key' -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")
[ "$CODE" = "401" ] || [ "$CODE" = "400" ] || warn "wrong key returned HTTP $CODE (expected 401)"

pass "laptop -> LiteLLM (auth) -> router -> vLLM works. Gateway key: $LITELLM_MASTER_KEY"
