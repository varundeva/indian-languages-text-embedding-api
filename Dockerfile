# --- Stage 1: Builder ---
FROM python:3.12-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 1. FORCE the CPU index exclusively for PyTorch to guarantee 0 bytes of CUDA
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest normally (it will use the Torch we just installed)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    transformers \
    numpy \
    psutil \
    psycopg2-binary

# 3. Explicitly nuke any stray NVIDIA packages that sometimes get pulled as dependencies
RUN pip freeze | grep nvidia | xargs -r pip uninstall -y

# 4. Save model ensuring ONLY the safetensors format is kept
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    model_id = 'ai4bharat/IndicBERTv2-MLM-Sam-TLM'; \
    AutoTokenizer.from_pretrained(model_id).save_pretrained('/app/model'); \
    AutoModel.from_pretrained(model_id).save_pretrained('/app/model', safe_serialization=True)"

# 5. Paranoid cleanup: delete older .bin model weights if they generated alongside safetensors
RUN find /app/model -name "pytorch_model.bin" -type f -delete
RUN find /opt/venv -type d -name "__pycache__" -exec rm -r {} +


# --- Stage 2: Runtime ---
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY --from=builder /app/model /app/model
COPY app.py .

RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Force fully offline mode
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000

CMD ["uvicorn", "app:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "2", \
    "--log-level", "info", \
    "--no-access-log"]