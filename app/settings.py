# app/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from dotenv import load_dotenv

# Ensure .env is loaded if present
load_dotenv()

class Settings(BaseSettings):
    # If provided, we’ll call an OpenAI-compatible endpoint (OpenAI, OpenRouter, Together, etc.)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None  # e.g., https://api.openai.com/v1
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_MODEL: str = "google/gemma-4-26b-a4b-it:free"
    TOGETHER_API_KEY: str | None = None

    # Server options
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # SSL for outbound API calls (set False only for local dev if certs fail)
    REQUESTS_VERIFY_SSL: bool = True
    REQUESTS_CA_BUNDLE: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
