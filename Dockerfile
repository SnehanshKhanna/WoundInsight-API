# ==============================================================================
# WoundInsight-API: Production Dockerfile for Google Cloud Run (CPU-Optimized)
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered streaming logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    API_DEVICE=cpu \
    PORT=8080 \
    HF_MODEL_REPO_ID=SnehanshKhanna/WoundInsight-models

# Install essential system dependencies:
# - curl: Container healthchecks & network debugging
# - libgl1, libglib2.0-0, libgomp1: Headless image processing & OpenMP multithreading
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set container working directory
WORKDIR /app

# Step 1: Install PyTorch CPU-only wheels first to keep image lightweight (~1.2 GB vs ~5 GB with CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 3: Copy application source code only (checkpoints are fetched from Hugging Face at runtime)
COPY app/ app/
COPY src/ src/

# Step 4: Create required ephemeral runtime storage and model directories
RUN mkdir -p \
    checkpoints/classification \
    checkpoints/segmentation \
    checkpoints/tissue_segmentation

# Default Cloud Run container port
EXPOSE 8080

# Container healthcheck using Cloud Run PORT environment variable fallback
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Launch production server: binds to 0.0.0.0 and dynamically respects $PORT from Cloud Run
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
