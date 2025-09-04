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
    OPENROUTER_MODEL: str = "meta-llama/llama-3.3-8b-instruct:free"
    TOGETHER_API_KEY: str | None = None

    # Local HF fallback model for generation when no OpenAI key set
    HF_LOCAL_MODEL: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # Server options
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Misc
    ENABLE_EXPLANATIONS: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
