#!/bin/bash
# Entrypoint for every vLLM pod (prefill / decode / single). Behaviour is driven
# entirely by env vars set in the StatefulSet, so the three configs of the
# ablation grid share one template.
set -euo pipefail

ARGS=(
  "$MODEL_ID"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0 --port 8000
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-num-seqs "$MAX_NUM_SEQS"
)
[ -n "${QUANTIZATION:-}" ] && ARGS+=(--quantization "$QUANTIZATION")

if [ "${ENABLE_PREFIX_CACHING:-true}" = "true" ]; then
  ARGS+=(--enable-prefix-caching)
else
  ARGS+=(--no-enable-prefix-caching)
fi

if [ -n "${KV_TRANSFER_CONFIG:-}" ]; then
  # LMCache tiers: HBM (vLLM) -> local CPU RAM -> Mooncake distributed store
  # (store pod DRAM, spilling to NVMe). local_hostname must be this pod's IP so
  # the Mooncake transfer engine advertises a reachable address.
  cat > /tmp/lmcache.yaml <<EOF
chunk_size: ${LMCACHE_CHUNK_SIZE}
local_cpu: true
max_local_cpu_size: ${LMCACHE_LOCAL_CPU_GB}
remote_url: "mooncakestore://${MOONCAKE_MASTER_HOST}:50051/"
pre_caching_hash_algorithm: sha256_cbor_64bit
extra_config:
  use_exists_sync: true
  save_chunk_meta: false
  local_hostname: "${POD_IP}"
  metadata_server: "http://${MOONCAKE_MASTER_HOST}:8080/metadata"
  protocol: "tcp"
  device_name: ""
  global_segment_size: $(( MOONCAKE_CLIENT_SEGMENT_GB * 1024 * 1024 * 1024 ))
  master_server_address: "${MOONCAKE_MASTER_HOST}:50051"
  local_buffer_size: 0
  mooncake_prefer_local_alloc: false
EOF
  echo "---- LMCache config ----"; cat /tmp/lmcache.yaml
  export LMCACHE_CONFIG_FILE=/tmp/lmcache.yaml
  export PYTHONHASHSEED=0   # identical chunk hashes across pods
  ARGS+=(--kv-transfer-config "$KV_TRANSFER_CONFIG")
fi

# shellcheck disable=SC2206
[ -n "${VLLM_EXTRA_ARGS:-}" ] && ARGS+=($VLLM_EXTRA_ARGS)

echo "role=${ROLE} pod=${HOSTNAME} ip=${POD_IP}"
echo "vllm serve ${ARGS[*]}"
exec vllm serve "${ARGS[@]}"
