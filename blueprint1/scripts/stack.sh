# shellcheck shell=bash
# Functions that put the inference plane into one of the ablation configs.
#
#   A  baseline: 1 vLLM pod (vllm-single), no prefix caching, no router
#   B  3 decode pods, prefix caching, cache-aware router (sglang or kv), no Mooncake
#   C  2 prefill + 3 decode pods, prefix caching + LMCache/Mooncake, kv_router P/D
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

LMCACHE_KV_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
[ "${ENABLE_KV_TIER_C:-true}" = "true" ] || LMCACHE_KV_CONFIG=""   # fallback: C without LMCache/Mooncake

# deploy_pool <pool-name> <role> <replicas> <gpumem> <gpucores> <prefix-caching> <kv-transfer-json> <config>
deploy_pool() {
  POOL="$1" ROLE="$2" REPLICAS="$3" GPUMEM="$4" GPUCORES="$5" \
  ENABLE_PREFIX_CACHING="$6" KV_TRANSFER_CONFIG="$7" BP1_CONFIG="$8" \
    apply_tpl "$BP1_ROOT/manifests/inference/vllm-statefulset.yaml"
}

scale_pool() {  # scale_pool <name> <replicas>  (no-op if it doesn't exist)
  kubectl -n inference get sts "$1" >/dev/null 2>&1 && kubectl -n inference scale sts "$1" --replicas="$2" >/dev/null || true
}

delete_routers() {
  kubectl -n inference delete deploy sgl-router kv-router --ignore-not-found >/dev/null
}

wait_pods_gone() {  # wait until no pods of the given pool exist (frees their HAMi slice)
  wait_until "pods of $1 to terminate" 300 bash -c \
    "[ -z \"\$(kubectl -n inference get pods -l pool=$1 -o name)\" ]"
}

deploy_pools() {
  local cfg="$1"
  case "$cfg" in
    A)
      scale_pool vllm-prefill 0; scale_pool vllm-decode 0
      wait_pods_gone vllm-prefill; wait_pods_gone vllm-decode
      deploy_pool vllm-single single 1 "$SINGLE_GPUMEM" "$SINGLE_GPUCORES" false "" A
      wait_rollout inference sts/vllm-single 1200
      ;;
    B)
      scale_pool vllm-single 0; scale_pool vllm-prefill 0
      wait_pods_gone vllm-single; wait_pods_gone vllm-prefill
      deploy_pool vllm-decode decode "$DECODE_REPLICAS" "$DECODE_GPUMEM" "$DECODE_GPUCORES" true "" B
      wait_rollout inference sts/vllm-decode 1200
      ;;
    C)
      scale_pool vllm-single 0
      wait_pods_gone vllm-single
      deploy_pool vllm-prefill prefill "$PREFILL_REPLICAS" "$PREFILL_GPUMEM" "$PREFILL_GPUCORES" true "$LMCACHE_KV_CONFIG" C
      deploy_pool vllm-decode decode "$DECODE_REPLICAS" "$DECODE_GPUMEM" "$DECODE_GPUCORES" true "$LMCACHE_KV_CONFIG" C
      wait_rollout inference sts/vllm-prefill 1200
      wait_rollout inference sts/vllm-decode 1200
      ;;
    *) die "unknown config '$cfg' (A|B|C)";;
  esac
}

deploy_router() {
  local cfg="$1" impl
  export ROUTER_DECODE_URLS ROUTER_PREFILL_URLS=""
  ROUTER_DECODE_URLS="$(pod_urls vllm-decode "$DECODE_REPLICAS")"
  kubectl apply -f "$BP1_ROOT/manifests/router/router-service.yaml" >/dev/null
  kubectl -n inference create configmap kv-router-code \
    --from-file="$BP1_ROOT/router/kv_router.py" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  delete_routers
  case "$cfg" in
    A) log "Config A has no router (clients hit vllm-single directly)"; return 0;;
    B) impl="$ROUTER_IMPL_B";;
    C) impl="kv"; ROUTER_PREFILL_URLS="$(pod_urls vllm-prefill "$PREFILL_REPLICAS")";;
  esac
  if [ "$impl" = "sglang" ]; then
    apply_tpl "$BP1_ROOT/manifests/router/sglang-router.yaml"
    wait_rollout inference deploy/sgl-router 600
  else
    apply_tpl "$BP1_ROOT/manifests/router/kv-router.yaml"
    wait_rollout inference deploy/kv-router 300
  fi
  wait_until "router service endpoints" 120 bash -c \
    "kubectl -n inference get endpoints router -o jsonpath='{.subsets[0].addresses[0].ip}' | grep -q ."
}

set_config() {
  local cfg="$1"
  log "Switching inference plane to config $cfg"
  deploy_pools "$cfg"
  deploy_router "$cfg"
  echo "$cfg" > "$BP1_ROOT/.active_config"
  kubectl -n inference get pods -o wide
}

# Endpoint (in-cluster URL) the benchmarks should target for a config.
endpoint_for() {
  case "$1" in
    A) echo "http://vllm-single-0.vllm-single.inference.svc.cluster.local:8000";;
    *) echo "http://router.inference.svc.cluster.local:8080";;
  esac
}

# /metrics URLs to snapshot for a config.
metric_targets() {
  case "$1" in
    A) pod_urls vllm-single 1;;
    B) pod_urls vllm-decode "$DECODE_REPLICAS";;
    C) echo "$(pod_urls vllm-prefill "$PREFILL_REPLICAS") $(pod_urls vllm-decode "$DECODE_REPLICAS") http://${MOONCAKE_MASTER_HOST}:9003";;
  esac
}
