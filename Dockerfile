# ============================================================
# LLM-Powered NLP Microservice — Multi-stage Docker Build
# ============================================================
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Pre-download models during build (caches in image)
RUN python -c "\
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration; \
pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english'); \
T5Tokenizer.from_pretrained('t5-small'); \
T5ForConditionalGeneration.from_pretrained('t5-small'); \
pipeline('ner', model='dslim/bert-base-NER'); \
print('Models downloaded successfully')"

EXPOSE 8000 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
