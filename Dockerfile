FROM python:3.11-slim-bookworm

# buildx sets TARGETARCH. CPU service — builds multi-arch in CI. Two deps differ
# by arch (both verified 2026-09-28): the x86 `torch==2.5.1+cpu` local tag has no
# aarch64 build (arm64 uses plain `torch==2.5.1` from the cpu index), and
# `tensorflow-cpu` is x86-only (arm64 uses `tensorflow`, which has aarch64 wheels).
ARG TARGETARCH

WORKDIR /app

# System deps for RDKit
RUN apt-get update && apt-get install -y \
    curl gcc g++ libxrender1 libxext6 libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (CPU) + Chemprop + TensorFlow (CPU) for all three predictors.
COPY requirements.txt .
RUN if [ "$TARGETARCH" = "arm64" ]; then TORCH="torch==2.5.1"; TF="tensorflow>=2.15,<2.16"; \
    else TORCH="torch==2.5.1+cpu"; TF="tensorflow-cpu>=2.15,<2.16"; fi && \
    pip install --no-cache-dir "$TORCH" --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "$TF" && \
    pip install --no-cache-dir --no-deps alfabet==0.4.1 nfp && \
    pip install --no-cache-dir pooch joblib pandas tqdm networkx

# Application code
COPY app/ app/
COPY main.py .

# Non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models && \
    chown -R appuser:appuser /app
USER appuser

ENV PORT=8030
ENV PYTHONUNBUFFERED=1
EXPOSE 8030

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8030/health || exit 1

CMD ["python", "main.py"]
