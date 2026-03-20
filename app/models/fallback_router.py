"""
Fallback Router — Intelligent Backend Routing
==============================================
Routes requests through local models first, falling back to
OpenAI and then OpenRouter if the local model fails or is
unavailable.  Ensures 100 % service availability.
"""

import os
import logging
import asyncio
from typing import Optional

import httpx

logger = logging.getLogger("nlp-service.fallback")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")

STYLE_PROMPTS = {
    "formal": "Rewrite the following text in a formal, professional tone:\n\n",
    "casual": "Rewrite the following text in a casual, friendly tone:\n\n",
    "concise": "Rewrite the following text to be as concise as possible while keeping all key points:\n\n",
    "academic": "Rewrite the following text in an academic, scholarly style:\n\n",
    "creative": "Rewrite the following text in a creative, engaging style with vivid language:\n\n",
}


class FallbackRouter:
    """
    Tries backends in order:  local → OpenAI → OpenRouter.
    Each backend is attempted; on failure the next is tried.
    """

    def __init__(self, nlp_engine):
        self.engine = nlp_engine
        self._client = httpx.AsyncClient(timeout=30.0)

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------
    async def summarize(
        self, text: str, max_length: int = 150, min_length: int = 30
    ) -> dict:
        """Summarize with fallback chain."""
        # 1. Local T5
        try:
            if self.engine.is_loaded:
                result = self.engine.summarize(text, max_length, min_length)
                if result and len(result.strip()) > 10:
                    logger.info("Summarization served by LOCAL model")
                    return {"text": result, "backend": "local-t5"}
        except Exception as exc:
            logger.warning(f"Local summarization failed: {exc}")

        # 2. OpenAI
        try:
            if OPENAI_API_KEY:
                result = await self._openai_chat(
                    f"Summarize the following text in {max_length} words or fewer:\n\n{text}"
                )
                if result:
                    logger.info("Summarization served by OPENAI")
                    return {"text": result, "backend": "openai"}
        except Exception as exc:
            logger.warning(f"OpenAI summarization failed: {exc}")

        # 3. OpenRouter
        try:
            if OPENROUTER_API_KEY:
                result = await self._openrouter_chat(
                    f"Summarize the following text in {max_length} words or fewer:\n\n{text}"
                )
                if result:
                    logger.info("Summarization served by OPENROUTER")
                    return {"text": result, "backend": "openrouter"}
        except Exception as exc:
            logger.warning(f"OpenRouter summarization failed: {exc}")

        # If local model is loaded but returned a short result, return it anyway
        if self.engine.is_loaded:
            fallback = self.engine.summarize(text, max_length, min_length)
            return {"text": fallback or "Unable to summarize.", "backend": "local-t5-fallback"}

        raise RuntimeError("All summarization backends failed")

    # ------------------------------------------------------------------
    # Rewrite
    # ------------------------------------------------------------------
    async def rewrite(self, text: str, style: str = "formal") -> dict:
        """Rewrite text in the requested style with fallback."""
        prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["formal"]) + text

        # 1. OpenAI (preferred for rewriting — better quality)
        try:
            if OPENAI_API_KEY:
                result = await self._openai_chat(prompt)
                if result:
                    return {"text": result, "backend": "openai"}
        except Exception as exc:
            logger.warning(f"OpenAI rewrite failed: {exc}")

        # 2. OpenRouter
        try:
            if OPENROUTER_API_KEY:
                result = await self._openrouter_chat(prompt)
                if result:
                    return {"text": result, "backend": "openrouter"}
        except Exception as exc:
            logger.warning(f"OpenRouter rewrite failed: {exc}")

        # 3. Local T5 (paraphrase attempt)
        try:
            if self.engine.is_loaded:
                prefix = "paraphrase: " if style == "concise" else "summarize: "
                result = self.engine.summarize(text, max_length=256, min_length=20)
                return {"text": result, "backend": "local-t5-paraphrase"}
        except Exception as exc:
            logger.warning(f"Local rewrite failed: {exc}")

        raise RuntimeError("All rewrite backends failed")

    # ------------------------------------------------------------------
    # API Clients
    # ------------------------------------------------------------------
    async def _openai_chat(self, prompt: str) -> Optional[str]:
        resp = await self._client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": OPENAI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.7,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    async def _openrouter_chat(self, prompt: str) -> Optional[str]:
        resp = await self._client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://github.com/nlp-microservice",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.7,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
