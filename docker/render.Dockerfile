FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python-is-python3 python3-dev git ca-certificates \
    build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-render.txt .
RUN python -m pip install --no-cache-dir setuptools==75.8.0 wheel==0.45.1 \
    && python -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 \
       --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir -r requirements-render.txt
ARG TORCH_CUDA_ARCH_LIST=12.0
ARG MAX_JOBS=4
ENV FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS}
# Same PyTorch3D source revision used to render the benchmark depth images.
RUN python -m pip install --no-cache-dir --no-build-isolation \
    'git+https://github.com/facebookresearch/pytorch3d.git@1f7f85c0a3abcce0644f33d32ee911f10a9cac26'
COPY cvxnet_inference/ cvxnet_inference/
COPY tests/ tests/
COPY infer.py model_config.json ./
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
ENTRYPOINT ["python", "-m", "cvxnet_inference.render_depth"]
