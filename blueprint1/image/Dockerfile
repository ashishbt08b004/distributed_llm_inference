# One image for the vLLM pods, the Mooncake master/store, the kv-router and the
# bench client: vLLM + LMCache (matched build) + Mooncake transfer engine/store.
ARG BASE_IMAGE=lmcache/vllm-openai:v0.5.5
FROM ${BASE_IMAGE}

# The base image keeps vLLM + LMCache in a uv-created venv at /opt/venv (first on
# PATH, so `python3` is the venv's). That venv has no pip, so a bare `pip` falls
# through to Ubuntu 24.04's system pip, which refuses (PEP 668,
# externally-managed-environment). Install into the interpreter that has vLLM:
# with uv when present (how the base image was built), else that interpreter's pip.
#
# check_cuda.py fails the build if vLLM and PyTorch target different CUDA majors
# (the v0.5.5-cu129 tag does; see that file).
#
# Mooncake ships one wheel per CUDA major (mooncake-transfer-engine for CUDA 12,
# mooncake-transfer-engine-cuda13 for 13), same `mooncake` module. Pick the one
# matching the image's PyTorch.
#
# The import checks need libcuda.so.1, which only exists at run time (the NVIDIA
# runtime mounts the host driver). For the checks alone, point the loader at the
# CUDA toolkit's stub libcuda in a throwaway directory. It is NOT left in the
# image, so it can never shadow the real driver in a running pod.
ARG MOONCAKE_PIP_VERSION=0.3.13.post1
COPY check_cuda.py /opt/bp1/check_cuda.py
RUN set -eux; \
    PY="$(command -v python3)"; \
    "$PY" -c "import vllm" ; \
    CUDA_MAJOR="$("$PY" /opt/bp1/check_cuda.py)"; \
    if [ "$CUDA_MAJOR" -ge 13 ]; then MOONCAKE_PKG=mooncake-transfer-engine-cuda13; else MOONCAKE_PKG=mooncake-transfer-engine; fi; \
    if command -v uv >/dev/null 2>&1; then \
      uv pip install --no-cache --python "$PY" "${MOONCAKE_PKG}==${MOONCAKE_PIP_VERSION}"; \
    else \
      "$PY" -m pip install --no-cache-dir "${MOONCAKE_PKG}==${MOONCAKE_PIP_VERSION}"; \
    fi; \
    command -v mooncake_master; \
    STUB="$(ls /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/targets/*/lib/stubs/libcuda.so 2>/dev/null | head -1 || true)"; \
    if [ -n "$STUB" ]; then \
      mkdir -p /tmp/cuda-stub && ln -sf "$STUB" /tmp/cuda-stub/libcuda.so.1; \
      LD_LIBRARY_PATH="/tmp/cuda-stub:${LD_LIBRARY_PATH:-}" \
        "$PY" -c "from mooncake.store import MooncakeDistributedStore; print('mooncake store OK')"; \
      LD_LIBRARY_PATH="/tmp/cuda-stub:${LD_LIBRARY_PATH:-}" \
        "$PY" -c "import vllm, lmcache, aiohttp; print('vllm', vllm.__version__)"; \
      rm -rf /tmp/cuda-stub; \
    else \
      echo "WARNING: no CUDA stub libcuda.so in the base image; skipping import checks (Phase 4's GPU check covers them)"; \
      "$PY" -c "import importlib.util as u; assert u.find_spec('mooncake.store'), 'mooncake.store missing'; print('mooncake installed')"; \
    fi

ENTRYPOINT []
