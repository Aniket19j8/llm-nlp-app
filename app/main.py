import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from .schemas import (
    SentimentRequest,
    SentimentResponse,
    RewriteRequest,
    RewriteResponse,
)
from .models import SentimentService, GeneratorService
from .metrics import REQUESTS, LATENCY
from .settings import settings
from .explain import TokenAttributor

app = FastAPI(title="LLM NLP App", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sentiment_svc = SentimentService()
writer_svc = GeneratorService()
explainer = TokenAttributor() if settings.ENABLE_EXPLANATIONS else None

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.post("/v1/sentiment", response_model=SentimentResponse)
def sentiment(req: SentimentRequest):
    REQUESTS.labels("/v1/sentiment").inc()
    start = time.time()
    label, score = sentiment_svc.predict(req.text)
    tokens = None
    if explainer:
        try:
            pairs = explainer.attribute(req.text)
            tokens = [{"token": t, "score": float(s)} for t, s in pairs]
        except Exception:
            tokens = None
    LATENCY.observe(time.time() - start)
    return SentimentResponse(label=label, score=score, tokens=tokens)

@app.post("/v1/rewrite", response_model=RewriteResponse)
def rewrite(req: RewriteRequest):
    REQUESTS.labels("/v1/rewrite").inc()
    start = time.time()
    text, latency_ms = writer_svc.rewrite(req.text, req.tone)
    LATENCY.observe(time.time() - start)
    return RewriteResponse(rewrite=text, latency_ms=latency_ms)
