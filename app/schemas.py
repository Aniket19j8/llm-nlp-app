from pydantic import BaseModel
from typing import List, Optional


class SentimentRequest(BaseModel):
    text: str


class TokenAttribution(BaseModel):
    token: str
    score: float


class SentimentResponse(BaseModel):
    label: str
    score: float
    tokens: Optional[List[TokenAttribution]] = None


class RewriteRequest(BaseModel):
    text: str
    tone: str = "friendly" # or "formal", "concise", etc.


class RewriteResponse(BaseModel):
    rewrite: str
    latency_ms: int