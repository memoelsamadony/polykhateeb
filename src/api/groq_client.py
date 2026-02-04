"""
Groq API client with model fallback support.
"""

import time
from typing import Dict, List, Optional, Any

from ..config import GROQ_API_KEY, GROQ_MODEL_FALLBACK

# Initialize Groq client
try:
    from groq import Groq
    import groq
    _GROQ_AVAILABLE = True
except ImportError:
    Groq = None
    groq = None
    _GROQ_AVAILABLE = False


def get_groq_client():
    """Get initialized Groq client or None if not available."""
    if not _GROQ_AVAILABLE or not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)


def _extract_retry_after_seconds(err: Exception) -> Optional[float]:
    """Extract retry-after seconds from error response headers."""
    try:
        resp = getattr(err, "response", None)
        if resp is not None:
            headers = getattr(resp, "headers", None) or {}
            ra = headers.get("retry-after") or headers.get("Retry-After")
            if ra is not None:
                return float(ra)
    except Exception:
        pass
    return None


def groq_chat_with_fallback(
    client,
    messages: List[Dict[str, str]],
    models: List[str] = None,
    *,
    temperature: float = 0.3,
    top_p: float = 1.0,
    max_tokens: int = 900,
    per_model_cooldown_sec: float = 10.0,
    hard_timeout_sec: float = 60.0,
) -> Optional[Dict[str, Any]]:
    """
    Call Groq API with automatic model fallback on rate limits.

    Args:
        client: Groq client instance
        messages: Chat messages
        models: List of model names to try (in priority order)
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        per_model_cooldown_sec: Cooldown after rate limit per model
        hard_timeout_sec: Total timeout for all attempts

    Returns:
        Dictionary with 'model' and 'text' keys, or None on failure
    """
    if models is None:
        models = GROQ_MODEL_FALLBACK

    cooldown_until: Dict[str, float] = {m: 0.0 for m in models}
    start = time.time()
    last_err: Optional[Exception] = None

    while (time.time() - start) < hard_timeout_sec:
        now = time.time()
        usable = [m for m in models if cooldown_until.get(m, 0.0) <= now]

        if not usable:
            soonest = min(cooldown_until.values())
            sleep_s = max(0.25, soonest - now)
            time.sleep(min(2.0, sleep_s))
            continue

        for model in usable:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stream=False,
                )
                return {"model": model, "text": resp.choices[0].message.content}

            except Exception as e:
                last_err = e
                is_rate_limit = False

                if _GROQ_AVAILABLE and groq:
                    if hasattr(groq, "RateLimitError") and isinstance(e, groq.RateLimitError):
                        is_rate_limit = True
                    if hasattr(groq, "APIStatusError") and isinstance(e, groq.APIStatusError):
                        status = getattr(e, "status_code", None) or getattr(e, "status", None)
                        if status == 429:
                            is_rate_limit = True

                if is_rate_limit:
                    ra = _extract_retry_after_seconds(e)
                    cooldown = ra if (ra is not None and ra > 0) else per_model_cooldown_sec
                    cooldown_until[model] = time.time() + cooldown
                    continue

                raise

    print(f"[CLOUD] Fallback timed out. Last error: {last_err}")
    return None
