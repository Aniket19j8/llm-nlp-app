import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import json
import requests

from .schemas import (
    SentimentRequest,
    SentimentResponse,
    RewriteRequest,
    RewriteResponse,
)
from .models import SentimentService, GeneratorService
from .metrics import REQUESTS, LATENCY

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


def _provider_error_detail(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            body = exc.response.json()
            return body.get("error", body) if isinstance(body, dict) else str(body)
        except Exception:
            return exc.response.text or str(exc)
    if isinstance(exc, requests.SSLError):
        return (
            "SSL certificate verification failed when calling the LLM API. "
            "For local dev, add REQUESTS_VERIFY_SSL=0 to your .env file."
        )
    if isinstance(exc, json.JSONDecodeError):
        return "LLM returned invalid JSON for sentiment. Try again."
    return str(exc)


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
    try:
        label, score = sentiment_svc.predict(req.text)
    except (requests.RequestException, json.JSONDecodeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=_provider_error_detail(exc)) from exc
    LATENCY.observe(time.time() - start)
    return SentimentResponse(label=label, score=score)


@app.post("/v1/rewrite", response_model=RewriteResponse)
def rewrite(req: RewriteRequest):
    REQUESTS.labels("/v1/rewrite").inc()
    start = time.time()
    try:
        text, latency_ms = writer_svc.rewrite(req.text, req.tone)
    except (requests.RequestException, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=_provider_error_detail(exc)) from exc
    LATENCY.observe(time.time() - start)
    return RewriteResponse(rewrite=text, latency_ms=latency_ms)
