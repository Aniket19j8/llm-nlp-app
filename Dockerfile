# ===== Build stage (deps + cache) =====
FROM python:3.11-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# Install torch (CPU) first from the official index, then the rest
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir "transformers==4.43.3" "tokenizers==0.19.1" "accelerate>=0.33.0" "huggingface-hub>=0.24.0" "safetensors>=0.4.4" "numpy<2"
COPY . .

# ===== Runtime =====
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY --from=base /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=base /usr/local/bin /usr/local/bin
COPY --from=base /app /app

# Hugging Face Spaces provides $PORT (default 7860). Bind to it.
EXPOSE 7860
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
