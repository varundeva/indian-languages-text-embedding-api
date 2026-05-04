# =============================================================================
# Stage 1: Model Conversion
# Uses torch ONLY to download and convert model to ONNX.
# Torch never appears in the final runtime image.
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

# torch only in builder — for model conversion
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# ONNX export tools — builder only
RUN pip install --no-cache-dir \
    transformers \
    "optimum[onnxruntime]" \
    onnx \
    onnxruntime

# Download + convert IndicBERT v2 → ONNX
RUN python - <<'PY'
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
model_id = 'ai4bharat/IndicBERTv2-MLM-Sam-TLM'
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
model.save_pretrained('/build/model')
AutoTokenizer.from_pretrained(model_id).save_pretrained('/build/model')
print('ONNX export complete')
PY

# Quantize ONNX to INT8 — cuts model size ~50% with minimal accuracy loss
RUN python - <<'PY'
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
src = '/build/model/model.onnx'
dst = '/build/model/model_q.onnx'
os.path.exists(src) and (quantize_dynamic(src, dst, weight_type=QuantType.QInt8), os.replace(dst, src))
print('Quantization complete')
PY

# Remove any leftover binary weights
RUN find /build/model -name "*.bin" -delete \
    && find /build/model -name "optimizer_*" -delete

# =============================================================================
# Stage 2: Runtime deps — no torch, no gcc, no build tools
# =============================================================================
FROM python:3.12-slim AS deps

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir \
    onnxruntime \
    transformers \
    fastapi \
    "uvicorn[standard]" \
    numpy \
    psutil

RUN find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
RUN find /opt/venv -name "*.pyc" -delete 2>/dev/null || true

# =============================================================================
# Stage 3: Final runtime image
# =============================================================================
FROM python:3.12-slim

WORKDIR /app

COPY --from=deps /opt/venv /opt/venv
COPY --from=builder /build/model /app/model
COPY app.py .

ENV PATH="/opt/venv/bin:$PATH" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    OMP_WAIT_POLICY=PASSIVE

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Single worker — ONNX releases GIL, handles concurrency fine
# Multiple workers would duplicate model in RAM (wasteful on small VPS)
CMD ["uvicorn", "app:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1", \
    "--log-level", "info", \
    "--no-access-log"]