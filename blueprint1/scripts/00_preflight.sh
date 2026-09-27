#!/usr/bin/env bash
# Phase 0 — verify the node (run ON the Lambda node).
source "$(dirname "$0")/lib.sh"

log "Installing small host tools (envsubst, jq, curl)"
sudo apt-get update -qq
sudo apt-get install -y -qq gettext-base jq curl ca-certificates >/dev/null

log "GPU / driver"
need nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
GPU_COUNT=$(nvidia-smi -L | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)

[[ "$GPU_NAME" == *H100* ]] || warn "GPU is '$GPU_NAME', blueprint assumes an H100"
(( GPU_COUNT == 1 )) || warn "Found $GPU_COUNT GPUs; blueprint assumes exactly 1"
(( MEM_MB > 79000 )) || warn "GPU memory ${MEM_MB} MB < 80 GB — shrink the slices in config.env"
(( PROCS == 0 )) || warn "$PROCS processes already on the GPU"

# The vLLM/LMCache image is built for CUDA 13 -> driver >= 580.
DRIVER_MAJOR=${DRIVER%%.*}
if (( DRIVER_MAJOR < 580 )); then
  warn "Driver $DRIVER is older than 580, which the ${BASE_IMAGE} image (CUDA 13) needs."
  warn "Pick the newest Lambda Stack image. (H100 may still run it via CUDA forward compatibility; Phase 4's GPU check will tell.)"
fi

log "Host resources"
free -g | head -2
df -h "$HOST_DATA_DIR" | tail -1
AVAIL_GB=$(df -BG --output=avail "$HOST_DATA_DIR" | tail -1 | tr -dc 0-9)
(( AVAIL_GB > 250 )) || warn "Only ${AVAIL_GB} GB free under $HOST_DATA_DIR (want >250 GB: image ~25 GB, model ~16 GB, Mooncake NVMe tier 200 GB)"

log "Container tooling"
if ! command -v docker >/dev/null; then
  warn "docker not found — installing docker.io (needed once to build the app image)"
  sudo apt-get install -y -qq docker.io >/dev/null
fi
sudo docker version --format 'docker {{.Server.Version}}'

[ -n "$HF_TOKEN" ] || warn "HF_TOKEN is not exported (Qwen2.5 is public, but you'll hit HF rate limits without it)"

mkdir -p "$HF_CACHE_DIR" "$VLLM_CACHE_DIR" "$MOONCAKE_DIR" "$BENCH_DIR/results"

if (( GPU_COUNT >= 1 && PROCS == 0 )); then
  pass "nvidia-smi shows ${GPU_NAME} (${MEM_MB} MB), driver ${DRIVER}, no processes running"
fi
