FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3
WORKDIR /app
COPY requirements-inference.txt .
RUN python -m pip install --no-cache-dir -r requirements-inference.txt
COPY vendor/ vendor/
COPY cvxnet_inference/ cvxnet_inference/
COPY tests/ tests/
COPY infer.py model_config.json ./
ENV PYTHONPATH=/app/vendor:/app TF_USE_LEGACY_KERAS=1 \
    TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
ENTRYPOINT ["python", "-m", "cvxnet_inference.predict"]
