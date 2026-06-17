# app/models.py
from __future__ import annotations

"""
LLM NLP App model services

- SentimentService:
    Sentiment classification via OpenAI-compatible chat APIs.

- GeneratorService:
    Rewrite text using the same provider chain with retries/backoff:
      1) OpenAI (primary, if configured)
      2) OpenRouter (optional fallback, supports free models)
      3) Together (optional fallback)
"""

import json
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import certifi
import requests

from .settings import settings

# Free OpenRouter models to try if the configured one is unavailable.
OPENROUTER_FREE_FALLBACKS = (
    "google/gemma-4-26b-a4b-it:free",
    "liquid/lfm-2.5-1.2b-instruct:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
)


def requests_verify() -> Union[bool, str]:
    if not settings.REQUESTS_VERIFY_SSL:
        return False
    if settings.REQUESTS_CA_BUNDLE:
        return settings.REQUESTS_CA_BUNDLE
    return certifi.where()


def _is_quota_exhausted(resp: requests.Response) -> bool:
    if resp.status_code != 429:
        return False
    try:
        err = resp.json().get("error", {})
        if isinstance(err, dict) and err.get("code") == "insufficient_quota":
            return True
    except Exception:
        pass
    return False


def build_providers() -> List["BaseChatProvider"]:
    providers: List[BaseChatProvider] = []

    openai_key = getattr(settings, "OPENAI_API_KEY", None)
    openai_base = getattr(settings, "OPENAI_BASE_URL", None)
    openai_org = getattr(settings, "OPENAI_ORG", None)
    if openai_key and openai_base:
        providers.append(OpenAIProvider(api_key=openai_key, base_url=openai_base, org=openai_org))

    or_key = getattr(settings, "OPENROUTER_API_KEY", None)
    or_model = getattr(settings, "OPENROUTER_MODEL", None) or OPENROUTER_FREE_FALLBACKS[0]
    if or_key:
        providers.append(OpenRouterProvider(api_key=or_key, model=or_model))

    tg_key = getattr(settings, "TOGETHER_API_KEY", None)
    if tg_key:
        providers.append(TogetherProvider(api_key=tg_key))

    if not providers:
        raise RuntimeError(
            "No chat providers configured. Set at least OPENAI_API_KEY and OPENAI_BASE_URL in .env, "
            "or OPENROUTER_API_KEY for OpenRouter-only."
        )
    return providers


def chat_with_fallback(
    providers: List["BaseChatProvider"],
    payload: Dict[str, Any],
    max_retries: int = 3,
) -> str:
    last_err: Optional[Exception] = None
    for provider in providers:
        try:
            return provider.chat(payload, max_retries=max_retries)
        except requests.HTTPError as e:
            last_err = e
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    raise RuntimeError("All providers failed without error?")


def _parse_sentiment_response(raw: str) -> Tuple[str, float]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    else:
        match = re.search(r"\{[^{}]*\"label\"[^{}]*\}", text)
        if match:
            text = match.group(0)

    data = json.loads(text)
    label = str(data["label"]).lower().strip()
    score = float(data["score"])
    if label not in {"positive", "negative", "neutral"}:
        raise ValueError(f"Unexpected sentiment label: {label}")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Sentiment score out of range: {score}")
    return label, score


# ------------------------ Chat provider base ------------------------
class BaseChatProvider:
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def _sleep_backoff(self, attempt: int, retry_after: Optional[str]) -> None:
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
            resp = requests.post(url, json=payload, headers=headers, timeout=60, verify=requests_verify())
            if resp.status_code < 400:
                return self._extract_text(resp.json())
            if _is_quota_exhausted(resp):
                resp.raise_for_status()
            if resp.status_code in (429, 503) and attempt < max_retries:
                self._sleep_backoff(attempt, resp.headers.get("Retry-After"))
                continue
            resp.raise_for_status()

        resp.raise_for_status()
        raise RuntimeError("OpenAI: unexpected")


class OpenRouterProvider(BaseChatProvider):
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1", model: Optional[str] = None) -> None:
        super().__init__("openrouter")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_override = model

    def _models_to_try(self, payload: Dict[str, Any]) -> List[str]:
        primary = self.model_override or payload.get("model")
        models: List[str] = []
        if primary:
            models.append(str(primary))
        for model in OPENROUTER_FREE_FALLBACKS:
            if model not in models:
                models.append(model)
        return models

    def chat(self, payload: Dict[str, Any], max_retries: int = 3) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "llm-nlp-app",
        }

        last_resp: Optional[requests.Response] = None
        for model in self._models_to_try(payload):
            pr_payload = dict(payload)
            pr_payload["model"] = model

            for attempt in range(max_retries + 1):
                resp = requests.post(url, json=pr_payload, headers=headers, timeout=60, verify=requests_verify())
                if resp.status_code < 400:
                    return self._extract_text(resp.json())

                last_resp = resp
                if resp.status_code in (400, 404):
                    break
                if resp.status_code in (429, 503) and attempt < max_retries:
                    self._sleep_backoff(attempt, resp.headers.get("Retry-After"))
                    continue
                resp.raise_for_status()

        if last_resp is not None:
            last_resp.raise_for_status()
        raise RuntimeError("OpenRouter: all models failed")


class TogetherProvider(BaseChatProvider):
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

        model_fallback = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        fallback_tried = False

        for attempt in range(max_retries + 1):
            resp = requests.post(url, json=tg_payload, headers=headers, timeout=60, verify=requests_verify())
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


class SentimentService:
    """Sentiment via OpenAI-compatible chat APIs."""

    def __init__(self) -> None:
        self.primary_model = getattr(settings, "OPENAI_MODEL", "gpt-4o-mini")
        self.providers = build_providers()

    def predict(self, text: str) -> Tuple[str, float]:
        payload: Dict[str, Any] = {
            "model": self.primary_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Classify the sentiment of the user's text. "
                        'Reply with ONLY valid JSON: {"label":"positive|negative|neutral","score":0.0-1.0}. '
                        "score is your confidence in the label."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "temperature": 0,
            "max_tokens": 64,
        }
        raw = chat_with_fallback(self.providers, payload)
        return _parse_sentiment_response(raw)


class GeneratorService:
    """Rewrite text via OpenAI -> OpenRouter -> Together."""

    def __init__(self) -> None:
        self.primary_model = getattr(settings, "OPENAI_MODEL", "gpt-4o-mini")
        self.providers = build_providers()

    def rewrite(self, text: str, tone: str) -> Tuple[str, int]:
        start = time.time()
        payload: Dict[str, Any] = {
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
            "max_tokens": 200,
        }
        content = chat_with_fallback(self.providers, payload)
        latency_ms = int((time.time() - start) * 1000)
        return content, latency_ms
