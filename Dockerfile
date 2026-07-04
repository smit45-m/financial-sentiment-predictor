# ─────────────────────────────────────────────────────────────
# Financial Sentiment Predictor — Dockerfile
# ─────────────────────────────────────────────────────────────
# Multi-purpose: supports both full (GPU-optional) and light
# deployments via build-args.
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Metadata
LABEL maintainer="financial-sentiment-predictor"
LABEL description="FastAPI sentiment analysis service"

# Prevent Python from writing .pyc files and enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --create-home appuser

# Working directory
WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY app/ ./app/
COPY src/ ./src/
COPY static/ ./static/
COPY saved_models/ ./saved_models/

# Ensure the non-root user owns the app directory
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE 8000

# Health check — verify the API is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
