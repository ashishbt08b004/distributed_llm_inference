"""Build-time guard: vLLM's compiled kernels and PyTorch must target the same CUDA major.

lmcache/vllm-openai:v0.5.5-cu129 installed an unpinned vLLM from PyPI, which is
built for CUDA 13, next to a CUDA 12.9 PyTorch. It imports fine without a GPU,
then dies at pod start with 'libcudart.so.13: cannot open shared object file'.
This reads the CUDA runtime each vLLM extension links (from the library names
inside the .so files), so it needs no GPU and fails the build instead.

Prints the torch CUDA major on success (used to pick the Mooncake wheel).
"""
import glob
import os
import re
import sys

import torch
import vllm

torch_major = torch.version.cuda.split(".")[0]
vllm_majors: set[str] = set()
for so in glob.glob(os.path.join(os.path.dirname(vllm.__file__), "**", "*.so"), recursive=True):
    with open(so, "rb") as f:
        vllm_majors.update(m.decode() for m in re.findall(rb"libcudart\.so\.(\d+)", f.read()))

print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), vllm {vllm.__version__} "
      f"links libcudart {sorted(vllm_majors) or '(none found)'}", file=sys.stderr)
if vllm_majors and vllm_majors != {torch_major}:
    sys.exit(f"BASE_IMAGE is inconsistent: vLLM is built for CUDA {sorted(vllm_majors)} but torch for "
             f"CUDA {torch_major}. Pick a base image where they match (see BASE_IMAGE in config.env).")
print(torch_major)
