# --- Stage 1: Builder ---
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies (discarded in final stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
# Make sure we use the venv pip
ENV PATH="/opt/venv/bin:$PATH"

# 1. Combine pip installs to prevent CUDA bloat
# 2. Use --extra-index-url so it applies to torch requirements from transformers
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn \
    transformers \
    numpy \
    psutil \
    psycopg2-binary \
    torch \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Clean up unneeded pycache files to shave off MBs
RUN find /opt/venv -type d -name "__pycache__" -exec rm -r {} +

# Download and save the model explicitly to avoid symlink duplication bloat
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    model_id = 'ai4bharat/IndicBERTv2-MLM-Sam-TLM'; \
    AutoTokenizer.from_pretrained(model_id).save_pretrained('/app/model'); \
    AutoModel.from_pretrained(model_id).save_pretrained('/app/model')"


# --- Stage 2: Runtime ---
FROM python:3.12-slim

WORKDIR /app

# Copy the completely built virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the explicitly saved model (no symlinks!)
COPY --from=builder /app/model /app/model

# Copy app code
COPY app.py .

# Create non-root user and set permissions
RUN useradd -m appuser && \
    chown -R appuser /app
USER appuser

# Force transformers to run fully offline so it doesn't attempt to download anything at runtime
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000

CMD ["uvicorn", "app:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "2", \
    "--log-level", "info", \
    "--no-access-log"]