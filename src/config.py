"""
Configuration and environment loading for PolyKhateeb.
"""

import os
from pathlib import Path

# Load .env file if present
_ENV_PATH = os.path.join(os.getcwd(), ".env")
if os.path.exists(_ENV_PATH):
    try:
        with open(_ENV_PATH, "r", encoding="utf-8") as _env:
            for _line in _env:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _v = _line.split("=", 1)
                _k = _k.strip()
                if _k and _k not in os.environ:
                    os.environ[_k] = _v.strip()
    except Exception as e:
        print(f"[ENV] Could not load .env: {e}")


# API Configuration
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "https://tbp7nv9wcm5joa-4000.proxy.runpod.net/llm/chat")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# LLM Headers
LLM_API_HEADERS = {"Content-Type": "application/json"}
if LLM_API_KEY:
    LLM_API_HEADERS["x-api-key"] = LLM_API_KEY

# Default LLM model
DEFAULT_MT_MODEL = "command-r:35b-8k"

# Groq fallback models (in priority order)
GROQ_MODEL_FALLBACK = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k2-instruct",
    "",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "groq/compound-mini",
]

# Islamic terminology glossary
GLOSSARY = {
    "الله": "Allah",
    "القرآن": "the Qur'an",
    "محمد": "Muhammad",
    "صلاة": "Salah",
    "زكاة": "Zakah",
    "جهاد": "Jihad",
    "تقوى": "Taqwa",
}

# Worker configuration
CONTEXT_WINDOW_SIZE = 15  # Max previous chunks kept for context

# Paths
LOGS_DIR = Path(os.getcwd()) / "logs"
LOGS_DIR.mkdir(exist_ok=True)
