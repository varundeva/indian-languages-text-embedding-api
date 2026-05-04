import os
import logging
import multiprocessing
import threading
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import psutil
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("indicbert-api")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR       = os.getenv("MODEL_DIR", "/app/model")
MAX_LENGTH      = int(os.getenv("MAX_LENGTH", "512"))
MAX_BATCH_SIZE  = int(os.getenv("MAX_BATCH_SIZE", "64"))
TEXT_PREVIEW    = int(os.getenv("TEXT_PREVIEW", "500"))
CPU_USAGE_CAP   = float(os.getenv("CPU_USAGE_CAP", "0.5"))

# ── CPU thread allocation — 50% of available cores ───────────────────────────
total_cores     = multiprocessing.cpu_count()
allowed_threads = max(1, int(total_cores * CPU_USAGE_CAP))

logger.info(f"CPU cores: {total_cores} | Allocated: {allowed_threads} ({int(CPU_USAGE_CAP*100)}% cap)")

# ── Supported languages ───────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "kn": "Kannada",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
    "ur": "Urdu",
    "en": "English",
}

# ── Model state ───────────────────────────────────────────────────────────────
class ModelState:
    tokenizer:  AutoTokenizer = None
    session:    ort.InferenceSession = None
    ready:      bool = False
    lock:       threading.Lock = threading.Lock()  # thread-safe inference

state = ModelState()

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading model from: {MODEL_DIR}")
    logger.info(f"RAM available: {psutil.virtual_memory().available / 1e9:.1f} GB")

    # ONNX session options — apply CPU cap here
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads  = allowed_threads        # parallel within one op
    sess_opts.inter_op_num_threads  = max(1, allowed_threads // 2)  # parallel between ops
    sess_opts.execution_mode        = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    model_path = os.path.join(MODEL_DIR, "model.onnx")
    if not os.path.exists(model_path):
        raise RuntimeError(f"model.onnx not found at {model_path}")

    state.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    state.session   = ort.InferenceSession(
        model_path,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    state.ready = True

    ram_used = psutil.Process().memory_info().rss / 1e9
    logger.info(f"Model ready | RAM used: {ram_used:.2f} GB | Threads: {allowed_threads}")

    yield

    logger.info("Shutting down")
    del state.session
    del state.tokenizer
    state.ready = False

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Indic Language Embedding API",
    description="Sentence embeddings for Indian regional languages — IndicBERT v2 via ONNX",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# ── Request / Response models ─────────────────────────────────────────────────
class EmbedRequest(BaseModel):
    text:     str = Field(..., min_length=1, max_length=5000)
    language: str = Field(..., description="ISO 639-1 code e.g. kn, hi, ta")

    @field_validator("language", mode="before")
    @classmethod
    def validate_language(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{v}'. "
                f"Supported: {sorted(SUPPORTED_LANGUAGES.keys())}"
            )
        return v

    @field_validator("text", mode="before")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text cannot be empty or whitespace")
        return v


class BatchEmbedItem(BaseModel):
    text:     str = Field(..., min_length=1, max_length=5000)
    language: str = Field(..., description="ISO 639-1 code")

    @field_validator("language", mode="before")
    @classmethod
    def validate_language(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{v}'")
        return v

    @field_validator("text", mode="before")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text cannot be empty")
        return v


class BatchEmbedRequest(BaseModel):
    items: List[BatchEmbedItem] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class EmbedResponse(BaseModel):
    embedding:  List[float]
    dimensions: int
    language:   str
    model:      str


class BatchEmbedResponse(BaseModel):
    embeddings: List[List[float]]
    count:      int
    dimensions: int
    model:      str


# ── Core inference ────────────────────────────────────────────────────────────
def mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    # last_hidden_state: [batch, seq_len, hidden]
    # attention_mask:    [batch, seq_len]
    mask        = attention_mask[:, :, np.newaxis].astype(np.float32)  # [batch, seq, 1]
    sum_emb     = (last_hidden_state * mask).sum(axis=1)               # [batch, hidden]
    sum_mask    = mask.sum(axis=1).clip(min=1e-9)                      # [batch, 1]
    return sum_emb / sum_mask                                           # [batch, hidden]


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-9)
    return vectors / norms


def compute_embeddings(texts: List[str]) -> np.ndarray:
    trimmed = [t[:TEXT_PREVIEW] for t in texts]

    encoded = state.tokenizer(
        trimmed,
        return_tensors="np",       # numpy directly — no torch tensor conversion
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
    )

    # Thread-safe ONNX inference
    with state.lock:
        outputs = state.session.run(
            output_names=["last_hidden_state"],
            input_feed={
                "input_ids":      encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
                "token_type_ids": encoded.get(
                    "token_type_ids",
                    np.zeros_like(encoded["input_ids"])
                ).astype(np.int64),
            }
        )

    last_hidden_state = outputs[0]  # [batch, seq_len, 768]
    vectors = mean_pool(last_hidden_state, encoded["attention_mask"])
    vectors = l2_normalise(vectors)

    return vectors  # [batch, 768] float32 numpy array


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    mem = psutil.virtual_memory()
    proc_ram = psutil.Process().memory_info().rss / 1e9
    return {
        "status":            "ok" if state.ready else "loading",
        "model_dir":         MODEL_DIR,
        "runtime":           "onnxruntime",
        "threads_used":      allowed_threads,
        "total_cpu_cores":   total_cores,
        "cpu_cap":           f"{int(CPU_USAGE_CAP * 100)}%",
        "ram_total_gb":      round(mem.total / 1e9, 1),
        "ram_available_gb":  round(mem.available / 1e9, 1),
        "ram_used_pct":      mem.percent,
        "process_ram_gb":    round(proc_ram, 2),
    }


@app.get("/languages")
def list_languages():
    return {
        "supported": [
            {"code": code, "name": name}
            for code, name in sorted(SUPPORTED_LANGUAGES.items())
        ]
    }


@app.post("/embed", response_model=EmbedResponse)
def embed_single(req: EmbedRequest):
    if not state.ready:
        raise HTTPException(status_code=503, detail="Model not ready — try again in 60s")

    try:
        vectors = compute_embeddings([req.text])
        return EmbedResponse(
            embedding=vectors[0].tolist(),
            dimensions=int(vectors.shape[1]),
            language=req.language,
            model=MODEL_DIR,
        )
    except Exception as e:
        logger.error(f"embed_single error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/batch", response_model=BatchEmbedResponse)
def embed_batch(req: BatchEmbedRequest):
    if not state.ready:
        raise HTTPException(status_code=503, detail="Model not ready — try again in 60s")

    try:
        texts   = [item.text for item in req.items]
        vectors = compute_embeddings(texts)
        return BatchEmbedResponse(
            embeddings=vectors.tolist(),
            count=int(vectors.shape[0]),
            dimensions=int(vectors.shape[1]),
            model=MODEL_DIR,
        )
    except Exception as e:
        logger.error(f"embed_batch error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))