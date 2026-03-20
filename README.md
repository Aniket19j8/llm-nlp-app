# 🧠 LLM-Powered NLP Microservice

A **production-grade GenAI platform** built with FastAPI, Docker, and Gradio, delivering text summarization, sentiment analysis, text rewriting, named entity recognition, and keyword extraction through transformer models (BERT/T5).

## ✨ Key Features

- **Multi-task NLP**: Summarization (T5), Sentiment (DistilBERT), NER (BERT-NER), Rewriting, Keywords
- **100% Availability**: Intelligent fallback routing — Local Models → OpenAI → OpenRouter
- **Sub-500ms Latency**: In-memory LRU cache with TTL + optimized inference
- **Blue-Green Deployment**: GitHub Actions CI/CD with automated smoke tests
- **Production Monitoring**: Request metrics, P95 latency tracking, health checks
- **Modular UI**: Gradio interface with tabs for each NLP capability

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────────────────────────────┐
│  Gradio UI  │────▶│          FastAPI Backend              │
│  (port 7860)│     │                                      │
└─────────────┘     │  ┌─────────┐  ┌──────────────────┐  │
                    │  │  Cache   │  │   Metrics         │  │
                    │  └────┬────┘  └──────────────────┘  │
                    │       │                              │
                    │  ┌────▼──────────────────────────┐  │
                    │  │      Fallback Router           │  │
                    │  │  Local T5/BERT → OpenAI → OR   │  │
                    │  └───────────────────────────────┘  │
                    └──────────────────────────────────────┘
```

## 🚀 Quick Start

### Docker (Recommended)
```bash
# Clone and start
git clone https://github.com/yourusername/llm-nlp-microservice.git
cd llm-nlp-microservice
cp .env.example .env  # Optional: add API keys for fallback
docker-compose up --build

# API: http://localhost:8000/docs
# UI:  http://localhost:7860
```

### Local Development
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Start API
uvicorn app.main:app --reload --port 8000

# Start UI (separate terminal)
python gradio_app.py
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check with metrics |
| POST | `/api/v1/summarize` | Text summarization |
| POST | `/api/v1/sentiment` | Sentiment analysis |
| POST | `/api/v1/rewrite` | Style-based text rewriting |
| POST | `/api/v1/ner` | Named entity recognition |
| POST | `/api/v1/keywords` | Keyword extraction |
| GET | `/api/v1/metrics` | Detailed service metrics |

### Example Request
```bash
curl -X POST http://localhost:8000/api/v1/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing!"}'
```

## 🧪 Testing
```bash
pytest tests/ -v
```

## 📁 Project Structure
```
├── app/
│   ├── main.py              # FastAPI application
│   ├── models/
│   │   ├── nlp_engine.py    # Local transformer models
│   │   └── fallback_router.py # Fallback routing logic
│   └── utils/
│       ├── cache.py         # LRU cache with TTL
│       └── metrics.py       # Request metrics collector
├── tests/
│   └── test_api.py          # Pytest test suite
├── gradio_app.py            # Gradio UI
├── Dockerfile               # Multi-stage Docker build
├── docker-compose.yml       # Docker Compose orchestration
├── .github/workflows/
│   └── ci-cd.yml            # Blue-green CI/CD pipeline
├── requirements.txt
└── README.md
```

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | No | Enables OpenAI fallback |
| `OPENROUTER_API_KEY` | No | Enables OpenRouter fallback |
| `OPENAI_MODEL` | No | Default: `gpt-3.5-turbo` |
| `OPENROUTER_MODEL` | No | Default: `mistralai/mistral-7b-instruct` |

## License

MIT
