# app/models.py
from __future__ import annotations

"""
LLM NLP App model services

- SentimentService:
    Local sentiment classification using PyTorch + Transformers (DistilBERT).
    No TensorFlow and no transformers.pipeline() to avoid optional backends.

- GeneratorService:
    Rewrite text using a provider chain with retries/backoff:
      1) OpenAI (primary, if configured)
      2) OpenRouter (optional fallback, supports free models)
      3) Together (optional fallback)
    The first provider that succeeds returns the rewrite.

Env vars typically used (read via app.settings.Settings, but accessed with getattr for safety):
  OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_ORG(optional)
  OPENROUTER_API_KEY, OPENROUTER_MODEL (e.g., meta-llama/llama-3.3-8b-instruct:free)
  TOGETHER_API_KEY
"""

import time
import random
from typing import Any, Dict, Tuple, Optional

import requests
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .settings import settings


# ------------------------ Sentiment (local) ------------------------
class SentimentService:
    """Local sentiment using DistilBERT (CPU)."""

    def __init__(self) -> None:
        self.model_id = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.model.eval()
        self.id2label = self.model.config.id2label  # e.g. {0:'NEGATIVE', 1:'POSITIVE'}

    @torch.inference_mode()
    def predict(self, text: str) -> Tuple[str, float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        score, idx = torch.max(probs, dim=-1)
        label = self.id2label[int(idx)].lower()
        return label, float(score)


# ------------------------ Chat provider base ------------------------
class BaseChatProvider:
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def _sleep_backoff(self, attempt: int, retry_after: Optional[str]) -> None:
        # Honor Retry-After if present; otherwise exponential backoff with jitter
        if retry_after:
            try:
                time.sleep(float(retry_after))
                return
            except Exception:
                pass
        base = 1.0
        time.sleep(base * (2 ** attempt) + random.uniform(0, 0.5))

    def _extract_text(self, resp_json: Dict[str, Any]) -> str:
        return resp_json["choices"][0]["message"]["content"].strip()

    def chat(self, payload: Dict[str, Any], max_retries: int = 3) -> str:
        raise NotImplementedError


# ------------------------ OpenAI provider ------------------------
class OpenAIProvider(BaseChatProvider):
    def __init__(self, api_key: str, base_url: str, org: Optional[str] = None) -> None:
        super().__init__("openai")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.org = org

    def chat(self, payload: Dict[str, Any], max_retries: int = 3) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.org:
            headers["OpenAI-Organization"] = self.org

        for attempt in range(max_retries + 1):
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            if resp.status_code < 400:
                return self._extract_text(resp.json())
            if resp.status_code in (429, 503) and attempt < max_retries:
                self._sleep_backoff(attempt, resp.headers.get("Retry-After"))
                continue
            resp.raise_for_status()

        # Should not reach here
        resp.raise_for_status()
        raise RuntimeError("OpenAI: unexpected")


# ------------------------ OpenRouter provider ------------------------
class OpenRouterProvider(BaseChatProvider):
    """
    OpenRouter is OpenAI-compatible and offers some free/cheap models.
    Example free model: meta-llama/llama-3.3-8b-instruct:free
    """

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1", model: Optional[str] = None) -> None:
        super().__init__("openrouter")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_override = model

    def chat(self, payload: Dict[str, Any], max_retries: int = 3) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Optional headers per OpenRouter etiquette
            "HTTP-Referer": "http://localhost",
            "X-Title": "llm-nlp-app",
        }

        pr_payload = dict(payload)
        if self.model_override:
            pr_payload["model"] = self.model_override

        fallback_tried = False
        for attempt in range(max_retries + 1):
            resp = requests.post(url, json=pr_payload, headers=headers, timeout=60)
            if resp.status_code < 400:
                return self._extract_text(resp.json())

            # If the requested model isn't available here, try a known free one once
            if resp.status_code == 400 and not fallback_tried:
                pr_payload = dict(pr_payload)
                pr_payload["model"] = "meta-llama/llama-3.3-8b-instruct:free"
                fallback_tried = True
                continue

            if resp.status_code in (429, 503) and attempt < max_retries:
                self._sleep_backoff(attempt, resp.headers.get("Retry-After"))
                continue

            resp.raise_for_status()

        resp.raise_for_status()
        raise RuntimeError("OpenRouter: unexpected")


# ------------------------ Together provider ------------------------
class TogetherProvider(BaseChatProvider):
    """
    Together API (paid). Uses OpenAI-compatible /chat/completions.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.together.xyz/v1", model: Optional[str] = None) -> None:
        super().__init__("together")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_override = model

    def chat(self, payload: Dict[str, Any], max_retries: int = 3) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        tg_payload = dict(payload)
        if self.model_override:
            tg_payload["model"] = self.model_override

        # Known broadly available Together model if 400
        model_fallback = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        fallback_tried = False

        for attempt in range(max_retries + 1):
            resp = requests.post(url, json=tg_payload, headers=headers, timeout=60)
            if resp.status_code < 400:
                return self._extract_text(resp.json())

            if resp.status_code == 400 and not fallback_tried:
                tg_payload = dict(payload)
                tg_payload["model"] = self.model_override or model_fallback
                fallback_tried = True
                continue

            if resp.status_code in (429, 503) and attempt < max_retries:
                self._sleep_backoff(attempt, resp.headers.get("Retry-After"))
                continue

            resp.raise_for_status()

        resp.raise_for_status()
        raise RuntimeError("Together: unexpected")


# ------------------------ Rewrite service with fallback chain ------------------------
class GeneratorService:
    """
    Try OpenAI -> OpenRouter -> Together with retry/backoff.
    Configure via .env (Settings):
      OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, (optional) OPENAI_ORG
      OPENROUTER_API_KEY, OPENROUTER_MODEL (optional)
      TOGETHER_API_KEY (optional)
    """

    def __init__(self) -> None:
        # Primary model name to request; provider may override to what they host
        self.primary_model = getattr(settings, "OPENAI_MODEL", "gpt-4o-mini")

        self.providers: list[BaseChatProvider] = []

        # 1) OpenAI primary (if configured)
        openai_key = getattr(settings, "OPENAI_API_KEY", None)
        openai_base = getattr(settings, "OPENAI_BASE_URL", None)
        openai_org = getattr(settings, "OPENAI_ORG", None)
        if openai_key and openai_base:
            self.providers.append(OpenAIProvider(api_key=openai_key, base_url=openai_base, org=openai_org))

        # 2) OpenRouter fallback (optional; can also be primary if OpenAI not set)
        or_key = getattr(settings, "OPENROUTER_API_KEY", None)
        or_model = getattr(settings, "OPENROUTER_MODEL", None) or "meta-llama/llama-3.3-8b-instruct:free"
        if or_key:
            self.providers.append(OpenRouterProvider(api_key=or_key, model=or_model))

        # 3) Together fallback (optional)
        tg_key = getattr(settings, "TOGETHER_API_KEY", None)
        if tg_key:
            self.providers.append(TogetherProvider(api_key=tg_key))

        if not self.providers:
            raise RuntimeError(
                "No chat providers configured. Set at least OPENAI_API_KEY and OPENAI_BASE_URL in .env, "
                "or OPENROUTER_API_KEY for OpenRouter-only."
            )

    def rewrite(self, text: str, tone: str) -> Tuple[str, int]:
        start = time.time()
        base_payload: Dict[str, Any] = {
            "model": self.primary_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You rewrite user text in a {tone} tone without adding or removing facts. "
                        f"Keep it concise and preserve meaning."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "temperature": 0.5,
            "max_tokens": 200,  # modest to reduce rate/usage pressure
        }

        last_err: Optional[Exception] = None
        for provider in self.providers:
            try:
                content = provider.chat(base_payload, max_retries=3)
                latency_ms = int((time.time() - start) * 1000)
                return content, latency_ms
            except requests.HTTPError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue

        if last_err:
            raise last_err
        raise RuntimeError("All providers failed without error?")
