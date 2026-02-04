"""
OLLAMA API client for local LLM inference.
"""

import re
import time
import requests
from typing import Dict, Optional

from ..config import OLLAMA_API_URL, LLM_API_HEADERS, DEFAULT_MT_MODEL
from ..utils import log


def robust_ollama_call(payload: dict, timeout: int = 30, retries: int = 1) -> Optional[str]:
    """
    Call the /llm/chat endpoint with a messages array.

    Args:
        payload: Request payload with messages, model, and options
        timeout: Request timeout in seconds
        retries: Number of retry attempts

    Returns:
        Cleaned response text or None on failure
    """
    messages = payload.get("messages", []) if isinstance(payload, dict) else []
    if not messages:
        log("[API] Missing messages")
        return None

    options = payload.get("options", {}) if isinstance(payload, dict) else {}
    max_tokens = options.get("max_tokens") or payload.get("max_tokens") or 2048
    temperature = options.get("temperature")
    if temperature is None:
        temperature = payload.get("temperature", 0.2)

    request_body = {
        "messages": messages,
        "model": payload.get("model", DEFAULT_MT_MODEL),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                OLLAMA_API_URL,
                headers=LLM_API_HEADERS,
                json=request_body,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            if not data.get("success"):
                raise ValueError(f"API success flag false: {data}")

            text = str(data.get("response", "")).strip()

            # Clean response
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text = re.sub(
                r'^(Here is|Sure|Translation|Corrected|Okay|I have translated).*?:',
                '', text, flags=re.IGNORECASE
            ).strip()
            text = re.sub(r'(Note:|P.S.|Analysis:).*', '', text, flags=re.IGNORECASE).strip()
            text = text.strip('"').strip("'")

            # Reject if contains Chinese characters (common hallucination)
            if re.search(r'[\u4e00-\u9fff]', text):
                return None

            return text
        except Exception as e:
            log(f"[API] Error: {e}")
        if attempt < retries:
            time.sleep(1)
    return None


def parse_tagged_response(text: str) -> Dict[str, str]:
    """
    Parse [Arabic] / [German] / [English] tagged response.

    Args:
        text: Raw response text with section tags

    Returns:
        Dictionary with fixed_ar, en, and de keys
    """
    result = {"fixed_ar": "", "en": "", "de": ""}
    if not text:
        return result

    # Match sections between [Tag] headers
    sections = re.split(r'\[(?:Arabic|German|English)\]', text)
    tags = re.findall(r'\[(Arabic|German|English)\]', text)

    for tag, content in zip(tags, sections[1:]):
        content = content.strip()
        if tag == "Arabic":
            result["fixed_ar"] = content
        elif tag == "German":
            result["de"] = content
        elif tag == "English":
            result["en"] = content

    return result
