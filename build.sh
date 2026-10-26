#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Override when building for a different GPU, e.g. CUDA_ARCH_LIST='8.0;9.0;12.0'.
arch="${CUDA_ARCH_LIST:-}"
if [[ -z "$arch" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')"
fi
if [[ -z "$arch" ]]; then
  echo 'Set CUDA_ARCH_LIST to your target GPU architecture (e.g. 12.0 for RTX 5090).' >&2
  exit 2
fi
echo "Building PyTorch3D for CUDA architecture(s): $arch"
docker build -f docker/render.Dockerfile -t cvxnet-inference-render:1 \
  --build-arg "TORCH_CUDA_ARCH_LIST=$arch" --build-arg "MAX_JOBS=${MAX_JOBS:-4}" .
docker build -f docker/inference.Dockerfile -t cvxnet-inference-model:1 .
