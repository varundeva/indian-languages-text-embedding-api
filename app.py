import os
import logging
import multiprocessing
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import psutil
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from transformers import AutoTokenizer, AutoModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("indicbert-api")

# ── Config from environment ───────────────────────────────────────────────────
MODEL_NAME      = os.getenv("MODEL_NAME", "/app/model")
MAX_LENGTH      = int(os.getenv("MAX_LENGTH", "512"))
MAX_BATCH_SIZE  = int(os.getenv("MAX_BATCH_SIZE", "64"))
TEXT_PREVIEW    = int(os.getenv("TEXT_PREVIEW", "500"))   # chars before tokenise
CPU_USAGE_CAP   = float(os.getenv("CPU_USAGE_CAP", "0.5"))  # 50% of cores

# ── CPU Optimisation ──────────────────────────────────────────────────────────
total_cores     = multiprocessing.cpu_count()
allowed_threads = max(1, int(total_cores * CPU_USAGE_CAP))

torch.set_num_threads(allowed_threads)
torch.set_num_interop_threads(max(1, allowed_threads // 2))
torch.set_grad_enabled(False)

logger.info(f"CPU cores total: {total_cores} | Threads allocated: {allowed_threads} (50%)")

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
    tokenizer = None
    model     = None
    ready     = False

state = ModelState()

# ── Lifespan — load model on startup ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading model: {MODEL_NAME}")
    logger.info(f"RAM available: {psutil.virtual_memory().available / 1e9:.1f} GB")

    state.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    state.model     = AutoModel.from_pretrained(MODEL_NAME)
    state.model.eval()
    state.ready     = True

    ram_used = psutil.Process().memory_info().rss / 1e9
    logger.info(f"Model loaded | RAM used by process: {ram_used:.2f} GB")
    logger.info(f"Threads: {allowed_threads}/{total_cores} cores (50% cap)")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down — releasing model")
    del state.model
    del state.tokenizer
    state.ready = False

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Indic Language Embedding API",
    description="Sentence embeddings for Indian regional languages using IndicBERT v2",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# ── Request / Response models ─────────────────────────────────────────────────
class EmbedRequest(BaseModel):
    text:     str  = Field(..., min_length=1, max_length=5000)
    language: str  = Field(..., description="ISO 639-1 language code e.g. kn, hi, ta")

    @validator("language")
    def validate_language(cls, v):
        v = v.strip().lower()
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{v}'. "
                f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
            )
        return v

    @validator("text")
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("text cannot be empty or whitespace")
        return v.strip()


class BatchEmbedItem(BaseModel):
    text:     str = Field(..., min_length=1, max_length=5000)
    language: str = Field(..., description="ISO 639-1 language code")

    @validator("language")
    def validate_language(cls, v):
        v = v.strip().lower()
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{v}'")
        return v

    @validator("text")
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("text cannot be empty")
        return v.strip()


class BatchEmbedRequest(BaseModel):
    items: List[BatchEmbedItem] = Field(..., min_items=1, max_items=MAX_BATCH_SIZE)


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


# ── Core embedding logic ──────────────────────────────────────────────────────
def mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    mask         = attention_mask.unsqueeze(-1).float()
    sum_emb      = (last_hidden_state * mask).sum(dim=1)
    sum_mask     = mask.sum(dim=1).clamp(min=1e-9)
    return sum_emb / sum_mask


def compute_embeddings(texts: List[str]) -> np.ndarray:
    # Pre-trim text — reduces memory before tokenisation
    trimmed = [t[:TEXT_PREVIEW] for t in texts]

    inputs = state.tokenizer(
        trimmed,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
    )

    outputs    = state.model(**inputs)
    vectors    = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])

    # L2 normalise — dot product becomes cosine similarity in pgvector
    vectors    = torch.nn.functional.normalize(vectors, p=2, dim=1)

    return vectors.detach().numpy()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    mem = psutil.virtual_memory()
    return {
        "status":           "ok" if state.ready else "loading",
        "model":            MODEL_NAME,
        "threads_used":     allowed_threads,
        "total_cpu_cores":  total_cores,
        "cpu_cap":          f"{int(CPU_USAGE_CAP * 100)}%",
        "ram_total_gb":     round(mem.total / 1e9, 1),
        "ram_available_gb": round(mem.available / 1e9, 1),
        "ram_used_pct":     mem.percent,
    }


@app.get("/languages")
def list_languages():
    return {
        "supported": [
            {"code": code, "name": name}
            for code, name in SUPPORTED_LANGUAGES.items()
        ]
    }


@app.post("/embed", response_model=EmbedResponse)
def embed_single(req: EmbedRequest):
    if not state.ready:
        raise HTTPException(status_code=503, detail="Model not ready yet")

    try:
        vectors = compute_embeddings([req.text])
        return EmbedResponse(
            embedding=vectors[0].tolist(),
            dimensions=len(vectors[0]),
            language=req.language,
            model=MODEL_NAME,
        )
    except Exception as e:
        logger.error(f"embed_single error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/batch", response_model=BatchEmbedResponse)
def embed_batch(req: BatchEmbedRequest):
    if not state.ready:
        raise HTTPException(status_code=503, detail="Model not ready yet")

    try:
        texts   = [item.text for item in req.items]
        vectors = compute_embeddings(texts)

        return BatchEmbedResponse(
            embeddings=vectors.tolist(),
            count=len(vectors),
            dimensions=vectors.shape[1],
            model=MODEL_NAME,
        )
    except Exception as e:
        logger.error(f"embed_batch error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))