"""
LLM-Powered NLP Microservice — Production FastAPI Application
=============================================================
Provides text summarization, sentiment analysis, text rewriting,
named entity recognition, and keyword extraction via transformer models.
Includes intelligent fallback routing: local models → OpenAI → OpenRouter.
"""

import os
import time
import logging
import hashlib
from typing import Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.models.nlp_engine import NLPEngine
from app.models.fallback_router import FallbackRouter
from app.utils.cache import ResponseCache
from app.utils.metrics import MetricsCollector

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("nlp-service")

# ---------------------------------------------------------------------------
# Globals initialised during lifespan
# ---------------------------------------------------------------------------
nlp_engine: Optional[NLPEngine] = None
fallback_router: Optional[FallbackRouter] = None
cache: Optional[ResponseCache] = None
metrics: Optional[MetricsCollector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown logic."""
    global nlp_engine, fallback_router, cache, metrics
    logger.info("🚀  Starting NLP service …")
    nlp_engine = NLPEngine()
    fallback_router = FallbackRouter(nlp_engine)
    cache = ResponseCache(max_size=2000, ttl_seconds=3600)
    metrics = MetricsCollector()
    logger.info("✅  All models loaded — service ready")
    yield
    logger.info("🛑  Shutting down NLP service")


app = FastAPI(
    title="LLM-Powered NLP Microservice",
    version="2.0.0",
    description=(
        "Production GenAI platform for summarization, sentiment analysis, "
        "text rewriting, NER, and keyword extraction with intelligent "
        "local-to-API fallback routing."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000, description="Input text")
    max_length: int = Field(default=150, ge=10, le=1024, description="Max output tokens")
    min_length: int = Field(default=30, ge=5, le=512, description="Min output tokens")

class RewriteRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    style: str = Field(
        default="formal",
        description="Target style: formal | casual | concise | academic | creative",
    )

class SentimentResponse(BaseModel):
    label: str
    score: float
    confidence: str
    all_scores: dict

class SummaryResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    backend: str

class RewriteResponse(BaseModel):
    rewritten: str
    style: str
    backend: str

class NERResponse(BaseModel):
    entities: list[dict]
    count: int

class KeywordsResponse(BaseModel):
    keywords: list[dict]

class HealthResponse(BaseModel):
    status: str
    uptime: float
    models_loaded: bool
    cache_size: int
    total_requests: int
    avg_latency_ms: float


# ---------------------------------------------------------------------------
# Middleware — request timing
# ---------------------------------------------------------------------------

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    if metrics:
        metrics.record_request(request.url.path, elapsed_ms)
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check with service metrics."""
    return HealthResponse(
        status="healthy",
        uptime=metrics.uptime if metrics else 0,
        models_loaded=nlp_engine is not None and nlp_engine.is_loaded,
        cache_size=cache.size if cache else 0,
        total_requests=metrics.total_requests if metrics else 0,
        avg_latency_ms=metrics.avg_latency if metrics else 0,
    )


@app.post("/api/v1/summarize", response_model=SummaryResponse, tags=["NLP"])
async def summarize(req: TextRequest):
    """Summarize text using T5/BART with fallback to API providers."""
    cache_key = _cache_key("summarize", req.text, req.max_length)
    if cached := cache.get(cache_key):
        return cached

    result = await fallback_router.summarize(
        req.text, max_length=req.max_length, min_length=req.min_length
    )
    summary_text = result["text"]

    response = SummaryResponse(
        summary=summary_text,
        original_length=len(req.text.split()),
        summary_length=len(summary_text.split()),
        compression_ratio=round(len(summary_text.split()) / max(len(req.text.split()), 1), 3),
        backend=result["backend"],
    )
    cache.set(cache_key, response)
    return response


@app.post("/api/v1/sentiment", response_model=SentimentResponse, tags=["NLP"])
async def sentiment(req: TextRequest):
    """Analyse sentiment using BERT-based classifier."""
    cache_key = _cache_key("sentiment", req.text)
    if cached := cache.get(cache_key):
        return cached

    result = nlp_engine.analyse_sentiment(req.text)
    response = SentimentResponse(**result)
    cache.set(cache_key, response)
    return response


@app.post("/api/v1/rewrite", response_model=RewriteResponse, tags=["NLP"])
async def rewrite(req: RewriteRequest):
    """Rewrite text in the requested style."""
    cache_key = _cache_key("rewrite", req.text, req.style)
    if cached := cache.get(cache_key):
        return cached

    result = await fallback_router.rewrite(req.text, style=req.style)
    response = RewriteResponse(
        rewritten=result["text"],
        style=req.style,
        backend=result["backend"],
    )
    cache.set(cache_key, response)
    return response


@app.post("/api/v1/ner", response_model=NERResponse, tags=["NLP"])
async def named_entity_recognition(req: TextRequest):
    """Extract named entities from text."""
    cache_key = _cache_key("ner", req.text)
    if cached := cache.get(cache_key):
        return cached

    entities = nlp_engine.extract_entities(req.text)
    response = NERResponse(entities=entities, count=len(entities))
    cache.set(cache_key, response)
    return response


@app.post("/api/v1/keywords", response_model=KeywordsResponse, tags=["NLP"])
async def keyword_extraction(req: TextRequest):
    """Extract key phrases / keywords from text."""
    cache_key = _cache_key("keywords", req.text)
    if cached := cache.get(cache_key):
        return cached

    keywords = nlp_engine.extract_keywords(req.text)
    response = KeywordsResponse(keywords=keywords)
    cache.set(cache_key, response)
    return response


@app.get("/api/v1/metrics", tags=["System"])
async def get_metrics():
    """Detailed service metrics."""
    return metrics.snapshot() if metrics else {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_key(*parts) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
