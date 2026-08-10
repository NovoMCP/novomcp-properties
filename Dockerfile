FROM python:3.11-slim-bullseye

WORKDIR /app

# System deps for RDKit
RUN apt-get update && apt-get install -y \
    curl gcc g++ libxrender1 libxext6 libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (CPU) + Chemprop + TensorFlow (CPU) for all three predictors
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "tensorflow-cpu>=2.15,<2.16" && \
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
