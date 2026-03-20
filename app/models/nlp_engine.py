"""
NLP Engine — Local Transformer Model Manager
=============================================
Loads and manages BERT (sentiment), T5 (summarization), and
supporting models for NER and keyword extraction.
"""

import logging
import re
from collections import Counter
from typing import Optional

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    T5Tokenizer,
    T5ForConditionalGeneration,
)

logger = logging.getLogger("nlp-service.engine")

# ---------------------------------------------------------------------------
# Model registry — name → HuggingFace ID
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
    "summarization": "t5-small",
    "ner": "dslim/bert-base-NER",
    "zero-shot": "facebook/bart-large-mnli",
}

CONFIDENCE_THRESHOLDS = {"HIGH": 0.85, "MEDIUM": 0.65}


class NLPEngine:
    """Manages local transformer models for multiple NLP tasks."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models: dict = {}
        self._tokenizers: dict = {}
        self._pipelines: dict = {}
        self.is_loaded = False
        self._load_models()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_models(self):
        """Load all models into memory."""
        try:
            logger.info("Loading sentiment model …")
            self._pipelines["sentiment"] = pipeline(
                "sentiment-analysis",
                model=MODEL_REGISTRY["sentiment"],
                device=-1,  # CPU
                top_k=None,
            )

            logger.info("Loading T5-small for summarization …")
            self._tokenizers["t5"] = T5Tokenizer.from_pretrained(
                MODEL_REGISTRY["summarization"]
            )
            self._models["t5"] = T5ForConditionalGeneration.from_pretrained(
                MODEL_REGISTRY["summarization"]
            ).to(self.device)

            logger.info("Loading NER model …")
            self._pipelines["ner"] = pipeline(
                "ner",
                model=MODEL_REGISTRY["ner"],
                aggregation_strategy="simple",
                device=-1,
            )

            self.is_loaded = True
            logger.info(f"✅  All models loaded on {self.device}")
        except Exception as exc:
            logger.error(f"Model loading failed: {exc}")
            self.is_loaded = False

    # ------------------------------------------------------------------
    # Sentiment Analysis
    # ------------------------------------------------------------------
    def analyse_sentiment(self, text: str) -> dict:
        """Return sentiment label, score, confidence band, and all scores."""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded")

        results = self._pipelines["sentiment"](text[:512])
        # `top_k=None` returns list of dicts for each label
        if isinstance(results[0], list):
            results = results[0]

        all_scores = {r["label"]: round(r["score"], 4) for r in results}
        top = max(results, key=lambda r: r["score"])

        if top["score"] >= CONFIDENCE_THRESHOLDS["HIGH"]:
            confidence = "HIGH"
        elif top["score"] >= CONFIDENCE_THRESHOLDS["MEDIUM"]:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return {
            "label": top["label"],
            "score": round(top["score"], 4),
            "confidence": confidence,
            "all_scores": all_scores,
        }

    # ------------------------------------------------------------------
    # Summarization (local T5)
    # ------------------------------------------------------------------
    def summarize(
        self, text: str, max_length: int = 150, min_length: int = 30
    ) -> str:
        """Summarize text using local T5 model."""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded")

        prefix = "summarize: "
        inputs = self._tokenizers["t5"].encode(
            prefix + text, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self._models["t5"].generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        return self._tokenizers["t5"].decode(summary_ids[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Named Entity Recognition
    # ------------------------------------------------------------------
    def extract_entities(self, text: str) -> list[dict]:
        """Extract named entities with deduplication and grouping."""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded")

        raw = self._pipelines["ner"](text[:1024])
        entities = []
        seen = set()
        for ent in raw:
            key = (ent["word"].strip(), ent["entity_group"])
            if key not in seen:
                seen.add(key)
                entities.append(
                    {
                        "text": ent["word"].strip(),
                        "label": ent["entity_group"],
                        "score": round(float(ent["score"]), 4),
                        "start": ent["start"],
                        "end": ent["end"],
                    }
                )
        return sorted(entities, key=lambda e: e["score"], reverse=True)

    # ------------------------------------------------------------------
    # Keyword Extraction (TF-based, no extra model needed)
    # ------------------------------------------------------------------
    def extract_keywords(self, text: str, top_n: int = 10) -> list[dict]:
        """Simple TF-IDF-inspired keyword extraction."""
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "and",
            "but", "or", "nor", "not", "so", "yet", "both", "either",
            "neither", "each", "every", "all", "any", "few", "more",
            "most", "other", "some", "such", "no", "only", "own", "same",
            "than", "too", "very", "just", "because", "if", "when",
            "where", "how", "what", "which", "who", "whom", "this",
            "that", "these", "those", "it", "its", "i", "me", "my",
            "we", "our", "you", "your", "he", "him", "his", "she", "her",
            "they", "them", "their", "about", "up", "out", "then",
        }
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        filtered = [w for w in words if w not in stop_words]
        counts = Counter(filtered)
        total = len(filtered) or 1

        keywords = [
            {"keyword": word, "score": round(count / total, 4), "frequency": count}
            for word, count in counts.most_common(top_n)
        ]
        return keywords
