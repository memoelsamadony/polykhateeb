import streamlit as st
import threading
import queue
import time
import subprocess
import wave
import tempfile
import re
import requests
import os
import uuid
import json
import gc
import datetime
import difflib
import numpy as np
import sounddevice as sd
from typing import List, Optional, Dict, Any, Tuple

from faster_whisper import WhisperModel
from streamlit.runtime.scriptrunner import add_script_run_ctx

from groq import Groq
import groq

# ==========================================
# 1) CONFIGURATION
# ==========================================

OLLAMA_API_URL = "https://ym13l6kahy4sna-4000.proxy.runpod.net/llm/generate"
LLM_API_HEADERS = {
    "x-api-key": "default_api_key_change_me",
    "Content-Type": "application/json",
}
DEFAULT_MT_MODEL = "qwen3:8b"

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

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

GROQ_MODEL_FALLBACK: List[str] = [
    "qwen/qwen3-32b",
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k2-instruct",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "groq/compound-mini",
]

PROMPT_TEMPLATES = {
    "Standard (Fusha/MSA)": {
        "fixer": """ROLE: Conservative Islamic ASR Restorer.
TASK: Restore Arabic khutbah text corrupted by speech-to-text phonetic mistakes.

RULES (MUST):
1) Output Arabic ONLY.
2) Do NOT paraphrase. Do NOT add or remove sentences.
3) Only fix obvious ASR confusions that make the phrase nonsensical in khutbah context
   (e.g., اتقوا vs اشتقوا, كتاب الله vs كتابنا, شر الأمور vs وصر الأمور, شريك vs سريك).
4) If unsure, keep the original words unchanged.
5) No English words, no commentary, no headings.""",

        "translator": """TASK: Translate Arabic to English.
STYLE: Formal, clear sermon-like English (not poetic, not archaic).

RULES (MUST):
1) Output ONLY the translation text (no intro, no notes).
2) Translate ONLY what is present in the Arabic input. Do NOT add missing khutbah phrases/verses.
3) If a phrase is garbled/unclear, write [unclear] rather than guessing.
4) Keep Islamic terms consistent (Allah, Qur'an, Salah, Zakah, Jihad, Taqwa, etc.).
5) If the input is not Arabic (e.g., German/English), output exactly: [non-arabic].""",

        "translator_de": """TASK: Translate Arabic to German.
STYLE: Formal, clear sermon-like German (not poetic or archaic).

RULES (MUST):
1) Output ONLY the translation text (no intro, no notes).
2) Translate ONLY what is present in the Arabic input. Do NOT add missing khutbah phrases/verses.
3) If a phrase is garbled/unclear, write [unklar] rather than guessing.
4) Keep Islamic terms consistent (Allah, Quran, Salah, Zakah, Jihad, Taqwa, etc.).
5) If the input is not Arabic (e.g., English/German), output exactly: [non-arabic]."""
    },

    "Egyptian (Masri)": {
        "fixer": """ROLE: Conservative ASR Restorer (Egyptian Arabic).
TASK: Clean Egyptian Arabic khutbah/speech transcription produced by ASR.

RULES (MUST):
1) Output Arabic ONLY.
2) Preserve DIALECT. Do NOT convert Masri into MSA.
3) Do NOT paraphrase. Do NOT add or remove sentences.
4) Only fix obvious ASR phonetic typos that break meaning.
5) LOANWORDS: If you see Arabized English (e.g., "كاميكلز"), keep it as-is (do NOT replace with a random Arabic word).
6) Never insert English words, no commentary, no headings.""",

        "translator": """TASK: Translate Egyptian Arabic to English.
STYLE: Natural, conversational English (clear, not slangy unless the Arabic is slangy).

RULES (MUST):
1) Output ONLY the translation (no intro, no notes).
2) Translate ONLY what is present. Do NOT add missing khutbah phrases/verses.
3) If a phrase is garbled/unclear, write [unclear] rather than guessing.
4) LOANWORDS: If you detect Arabized English, translate it to the intended English term when confident; otherwise keep it as a transliteration in brackets.
5) If the input is not Arabic, output exactly: [non-arabic].""",

        "translator_de": """TASK: Translate Egyptian Arabic to German.
STYLE: Natural, conversational German (clear, not slangy unless the Arabic is slangy).

RULES (MUST):
1) Output ONLY the translation (no intro, no notes).
2) Translate ONLY what is present. Do NOT add missing khutbah phrases/verses.
3) If a phrase is garbled/unclear, write [unklar] rather than guessing.
4) LOANWORDS: If you detect Arabized English, translate it to the intended English term when confident; otherwise keep it as a transliteration in brackets.
5) If the input is not Arabic, output exactly: [non-arabic]."""
    },

    "Gulf (Khaleeji)": {
        "fixer": """ROLE: Conservative ASR Restorer (Gulf Arabic).
TASK: Clean Khaleeji Arabic transcription produced by ASR.

RULES (MUST):
1) Output Arabic ONLY.
2) Preserve DIALECT (Khaleeji). Do NOT convert to MSA.
3) Do NOT paraphrase. Do NOT add or remove sentences.
4) Only fix obvious ASR phonetic typos that break meaning.
5) Keep Khaleeji words as-is (e.g., شلونك، زين، علومك).
6) LOANWORDS: Handle English product/system terms correctly if present; do NOT replace with random Arabic.
7) Never insert English words, no commentary, no headings.""",

        "translator": """TASK: Translate Gulf Arabic to English.
STYLE: Respectful, natural English.

RULES (MUST):
1) Output ONLY the translation (no intro, no notes).
2) Translate ONLY what is present. Do NOT add missing khutbah phrases/verses.
3) If a phrase is garbled/unclear, write [unclear] rather than guessing.
4) Keep Islamic terms consistent (Allah, Qur'an, Salah, etc.).
5) If the input is not Arabic, output exactly: [non-arabic].""",

        "translator_de": """TASK: Translate Gulf Arabic to German.
STYLE: Respectful, natural German.

RULES (MUST):
1) Output ONLY the translation (no intro, no notes).
2) Translate ONLY what is present. Do NOT add missing khutbah phrases/verses.
3) If a phrase is garbled/unclear, write [unklar] rather than guessing.
4) Keep Islamic terms consistent (Allah, Quran, Salah, etc.).
5) If the input is not Arabic, output exactly: [non-arabic]."""
    }
}

GLOSSARY = {
    "الله": "Allah",
    "القرآن": "the Qur'an",
    "محمد": "Muhammad",
    "صلاة": "Salah",
    "زكاة": "Zakah",
    "جهاد": "Jihad",
    "تقوى": "Taqwa",
}

# ==========================================
# 2) LOGGING + PATHS
# ==========================================

def get_current_ts_string():
    now = datetime.datetime.now()
    return now.strftime("%H:%M:%S.%f")[:-3]

def log(msg):
    print(f"[{get_current_ts_string()}] {msg}", flush=True)

def get_log_path(filename):
    base_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, filename)

def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "")
    cleaned = cleaned.strip("._")
    return cleaned or "unnamed"

def append_to_file(filepath, text):
    if not text:
        return
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception as e:
        log(f"FILE ERROR ({filepath}): {e}")

# ==========================================
# 3) ARABIC UTILITIES + SPELLCHECK (PLUGGABLE)
# ==========================================

_AR_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_TATWEEL = "\u0640"

# Common khutbah-specific ASR confusions (very safe + high ROI)
KHUTBAH_FIXES = [
    (r"\bأشاد\b", "أشهد"),
    (r"\bسريك\b", "شريك"),
    (r"\bحيال\b", "حي"),
    (r"\bحي\b\s+الصلاة\b", "حي على الصلاة"),
    (r"\bحي\b\s+الفلاح\b", "حي على الفلاح"),
    (r"\bملغ\b\s+يوم\s+الدين\b", "مالك يوم الدين"),
    (r"\bوبجر\b\s+منهما\b", "وبثّ منهما"),
    (r"\bألعمت\b", "أنعمت"),
    (r"\bالمخضوب\b", "المغضوب"),
    (r"\bالارحام\b", "الأرحام"),
]

def arabic_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    ar_letters = sum(1 for c in letters if _AR_RE.match(c))
    return ar_letters / max(1, len(letters))

def collapse_elongation(s: str) -> str:
    # Collapse: حييييي -> حي, الللل -> ل
    return re.sub(r"(.)\1{4,}", r"\1", s or "")

def normalize_ar_basic(s: str) -> str:
    t = (s or "").replace(_TATWEEL, "")
    t = collapse_elongation(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def normalize_khutbah_staples(ar: str) -> str:
    t = normalize_ar_basic(ar)
    for pat, rep in KHUTBAH_FIXES:
        t = re.sub(pat, rep, t)
    return t

def repetition_score_words(words: List[str]) -> float:
    if not words:
        return 0.0
    from collections import Counter
    c = Counter(words)
    return c.most_common(1)[0][1] / max(1, len(words))

def nonword_rate(words: List[str]) -> float:
    # crude but fast: "word" must contain at least one Arabic char
    if not words:
        return 1.0
    good = sum(1 for w in words if _AR_RE.search(w))
    return 1.0 - (good / max(1, len(words)))

def is_gibberish_arabic(text: str) -> bool:
    t = normalize_ar_basic(text)
    if not t:
        return True
    if _LATIN_RE.search(t):
        return True
    if arabic_ratio(t) < 0.70:
        return True

    words = re.findall(r"\S+", t)
    if len(words) >= 12 and repetition_score_words(words) > 0.30:
        return True
    if len(words) >= 8 and nonword_rate(words) > 0.35:
        return True
    if re.search(r"(نحن\s+){4,}", t):
        return True
    return False

def glossary_block() -> str:
    lines = [f"- {ar} => {en}" for ar, en in GLOSSARY.items()]
    return "GLOSSARY (must follow exactly):\n" + "\n".join(lines)

def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()

# ---- Optional SymSpell-based Arabic spell-check (fast, offline, controllable) ----
# You can provide a dictionary file with one of these env vars:
#   AR_SYMSPELL_DICT=/path/to/ar_freq.txt
# Expected format (SymSpell): "term count" per line, UTF-8.
# Example:
#   الله 999999
#   القرآن 500000
#   الحمد 400000
# If no dict is provided, the spellcheck falls back to only KHUTBAH_FIXES + elongation collapse.

_SYMSPELL = None
try:
    from symspellpy import SymSpell, Verbosity  # type: ignore
    _SYMSPELL = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
except Exception:
    _SYMSPELL = None

def _load_symspell_dict_once() -> bool:
    if _SYMSPELL is None:
        return False
    if getattr(_SYMSPELL, "_dict_loaded", False):
        return True

    dict_path = os.getenv("AR_SYMSPELL_DICT", "").strip()
    if not dict_path:
        # also try local default
        maybe = os.path.join(os.getcwd(), "arabic_frequency_dictionary.txt")
        if os.path.exists(maybe):
            dict_path = maybe

    if not dict_path or not os.path.exists(dict_path):
        _SYMSPELL._dict_loaded = False
        return False

    try:
        # term_index=0, count_index=1
        ok = _SYMSPELL.load_dictionary(dict_path, term_index=0, count_index=1)
        _SYMSPELL._dict_loaded = bool(ok)
        log(f"[SPELL] SymSpell dict loaded: {dict_path} (ok={ok})")
        return bool(ok)
    except Exception as e:
        log(f"[SPELL] Failed to load SymSpell dict: {e}")
        _SYMSPELL._dict_loaded = False
        return False

def spellcheck_arabic(text: str, *, enable_symspell: bool = True) -> str:
    """
    Conservative spellcheck:
    1) normalize + khutbah staples fixes (safe)
    2) optional SymSpell for word-level corrections (only Arabic tokens, minimal edits)
    """
    t0 = normalize_khutbah_staples(text)

    if not enable_symspell or _SYMSPELL is None or not _load_symspell_dict_once():
        return t0

    # Protect glossary Arabic terms (do NOT change them)
    protected = set(GLOSSARY.keys())
    tokens = re.findall(r"\S+|\n", t0)  # preserve newlines
    out = []

    for tok in tokens:
        if tok == "\n":
            out.append(tok)
            continue

        w = tok
        # skip tokens containing punctuation mixed heavily
        if not _AR_RE.search(w):
            out.append(w)
            continue

        # strip light punctuation around
        lead = re.match(r"^\W+", w)
        trail = re.search(r"\W+$", w)
        pre = lead.group(0) if lead else ""
        suf = trail.group(0) if trail else ""
        core = re.sub(r"^\W+|\W+$", "", w)

        if core in protected or len(core) <= 3:
            out.append(pre + core + suf)
            continue

        # If core is mostly Arabic, attempt correction
        if arabic_ratio(core) < 0.8:
            out.append(pre + core + suf)
            continue

        try:
            suggestions = _SYMSPELL.lookup(core, Verbosity.TOP, max_edit_distance=2)
            if suggestions:
                best = suggestions[0].term
                # Very conservative: accept only if close and doesn't reduce Arabic-ness
                if best and best != core and abs(len(best) - len(core)) <= 2 and arabic_ratio(best) >= arabic_ratio(core):
                    core = best
        except Exception:
            pass

        out.append(pre + core + suf)

    return "".join(out).replace("  ", " ").strip()

# ==========================================
# 4) LLM CALLS
# ==========================================

def robust_ollama_call(payload, timeout=30, retries=1):
    messages = payload.get("messages", []) if isinstance(payload, dict) else []
    prompt_parts = []

    for m in messages:
        try:
            content = str(m.get("content", "")).strip()
            role = str(m.get("role", "")).strip().lower()
        except Exception:
            continue

        if not content:
            continue
        if role == "system":
            prompt_parts.append(content)
        else:
            prompt_parts.append(f"{role}: {content}" if role else content)

    prompt = "\n\n".join([p for p in prompt_parts if p]) or str(payload.get("prompt", "")).strip()
    if not prompt:
        log("[API] Missing prompt content")
        return None

    options = payload.get("options", {}) if isinstance(payload, dict) else {}
    max_tokens = options.get("max_tokens") or payload.get("max_tokens") or 512
    temperature = options.get("temperature")
    if temperature is None:
        temperature = payload.get("temperature", 0.3)

    request_body = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
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

            # Cleaner
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            text = re.sub(r"^(Here is|Sure|Translation|Corrected|Okay|I have translated).*?:", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"(Note:|P\.S\.|Analysis:).*", "", text, flags=re.IGNORECASE).strip()
            text = text.strip('"').strip("'").strip()

            # reject CJK
            if re.search(r"[\u4e00-\u9fff]", text):
                return None

            return text

        except Exception as e:
            log(f"[API] Error: {e}")
        if attempt < retries:
            time.sleep(1)
    return None

def _extract_retry_after_seconds(err: Exception) -> Optional[float]:
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
    client: Groq,
    messages: List[Dict[str, str]],
    models: List[str],
    *,
    temperature: float = 0.3,
    top_p: float = 1.0,
    max_tokens: int = 900,
    per_model_cooldown_sec: float = 10.0,
    hard_timeout_sec: float = 60.0,
) -> Optional[Dict[str, Any]]:
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

# ==========================================
# 5) WHISPER MODEL LOAD/UNLOAD (GPU DELOAD)
# ==========================================

@st.cache_resource(max_entries=1)
def load_whisper_model_cached(model_size, device, compute_type):
    id_map = {
        "distil-large-v3": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "large-v3": "Systran/faster-whisper-large-v3",
        "medium": "Systran/faster-whisper-medium",
    }
    hf_id = id_map.get(model_size, model_size)
    log(f"Loading Whisper: {hf_id} (Device: {device}, Type: {compute_type})...")
    gc.collect()
    m = WhisperModel(hf_id, device=device, compute_type=compute_type)
    log("Whisper Loaded Successfully.")
    return m

def unload_whisper_model_best_effort():
    """
    Best-effort GPU memory release:
    - clear streamlit cache_resource for the Whisper model
    - remove references
    - gc + cuda empty_cache
    """
    try:
        load_whisper_model_cached.clear()
    except Exception:
        pass

    # also clear any session reference
    try:
        st.session_state.whisper_model = None
    except Exception:
        pass

    gc.collect()

    # torch cache (even if CT2 is used, this is harmless)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

# ==========================================
# 6) WORKERS
# ==========================================

def cloud_polish_worker(
    cloud_in_q: "queue.Queue",
    cloud_out_q: "queue.Queue",
    *,
    flush_every: int = 8,
    session_uid: str = "",
    enable_spellcheck: bool = True,
):
    if not groq_client:
        log("[CLOUD] GROQ_API_KEY not set => cloud disabled.")
        while True:
            x = cloud_in_q.get()
            if x == "STOP":
                return

    buf = []
    start_id = None
    end_id = None
    last_flush = time.time()

    def make_system_prompt() -> str:
        return (
            "ROLE: Conservative Arabic ASR fixer and translator for khutbah/speech.\n"
            "TASK: Given raw Arabic ASR text (may contain multiple chunks separated by blank lines), do:\n"
            "1) Minimal Arabic fix (no paraphrase, no adding/removing sentences). Arabic only.\n"
            "2) Translate to English (en) and German (de).\n"
            "OUTPUT: Strict JSON with keys fixed_ar, en, de.\n"
            "{\"fixed_ar\": \"...\", \"en\": \"...\", \"de\": \"...\"}\n"
            "RULES:\n"
            "- fixed_ar must stay in Arabic and keep structure; fix only obvious ASR mistakes.\n"
            "- en/de must translate only what is present; if unclear, use [unclear] / [unklar].\n"
            "- If input is not Arabic, set fixed_ar='[non-arabic]' and en/de='[non-arabic]'.\n"
            "- No extra text, JSON only."
        )

    while True:
        try:
            item = cloud_in_q.get(timeout=0.5)
        except queue.Empty:
            item = None

        if item == "STOP":
            break

        time_due = (time.time() - last_flush) > 8.0

        if item:
            ar = (item.get("ar") or "").strip()
            rg = item.get("range")

            if ar:
                # ✅ spellcheck BEFORE sending to Groq cloud
                ar = spellcheck_arabic(ar, enable_symspell=enable_spellcheck)
                ar = normalize_khutbah_staples(ar)

                if rg and isinstance(rg, (tuple, list)) and len(rg) == 2:
                    if start_id is None:
                        start_id = int(rg[0])
                    end_id = int(rg[1])
                else:
                    if start_id is None:
                        start_id = int(item.get("id", 0))
                    end_id = int(item.get("id", 0))

                buf.append(ar)

            should_flush = (len(buf) >= flush_every) or bool(item.get("final", False)) or time_due
        else:
            should_flush = time_due

        if not should_flush or not buf:
            continue

        end_id = int(end_id) if end_id is not None else (start_id + len(buf) - 1 if start_id is not None else 0)
        batch = "\n\n".join(buf).strip()

        if not batch:
            buf = []
            start_id = None
            end_id = None
            last_flush = time.time()
            continue

        # If batch becomes gibberish, don't waste tokens
        if is_gibberish_arabic(batch):
            cloud_out_q.put({
                "range": (start_id or 0, end_id),
                "model": "cloud",
                "text": "[غير واضح / ASR hallucination]",
                "lang": "ar-fixed",
            })
            cloud_out_q.put({
                "range": (start_id or 0, end_id),
                "model": "cloud",
                "text": "[unclear]",
                "lang": "en",
            })
            cloud_out_q.put({
                "range": (start_id or 0, end_id),
                "model": "cloud",
                "text": "[unklar]",
                "lang": "de",
            })
            buf = []
            start_id = None
            end_id = None
            last_flush = time.time()
            continue

        log(f"[CLOUD] Flushing {start_id}-{end_id} ({len(buf)} parts) -> en/de with Arabic fix")

        messages = [
            {"role": "system", "content": make_system_prompt()},
            {"role": "user", "content": batch},
        ]

        result = groq_chat_with_fallback(
            groq_client,
            messages,
            GROQ_MODEL_FALLBACK,
            temperature=0.2,
            top_p=1.0,
            max_tokens=1500,
            per_model_cooldown_sec=10.0,
            hard_timeout_sec=60.0,
        )

        def _extract_json(txt: str) -> Dict[str, str]:
            try:
                t = (txt or "").strip()
                if t.startswith("```"):
                    t = re.sub(r"^```(json)?", "", t, flags=re.IGNORECASE).strip()
                    t = t.rstrip("`").rstrip()
                return json.loads(t)
            except Exception:
                return {}

        if result and result.get("text"):
            parsed = _extract_json(result["text"].strip())
            fixed_ar = str(parsed.get("fixed_ar", "")).strip() if parsed else ""
            en_text = str(parsed.get("en", "")).strip() if parsed else ""
            de_text = str(parsed.get("de", "")).strip() if parsed else ""

            if fixed_ar:
                cloud_out_q.put({"range": (start_id or 0, end_id), "model": result["model"], "text": fixed_ar, "lang": "ar-fixed"})
            if en_text:
                cloud_out_q.put({"range": (start_id or 0, end_id), "model": result["model"], "text": en_text, "lang": "en"})
            if de_text:
                cloud_out_q.put({"range": (start_id or 0, end_id), "model": result["model"], "text": de_text, "lang": "de"})

        buf = []
        start_id = None
        end_id = None
        last_flush = time.time()

def refinement_worker(
    input_q: "queue.Queue",
    output_q: "queue.Queue",
    cloud_q: Optional["queue.Queue"] = None,
    *,
    enable_spellcheck: bool = True,
):
    log("Refinement Worker: STARTING")

    current_fixer_prompt = PROMPT_TEMPLATES["Standard (Fusha/MSA)"]["fixer"]
    current_translator_en_prompt = PROMPT_TEMPLATES["Standard (Fusha/MSA)"]["translator"]
    current_translator_de_prompt = PROMPT_TEMPLATES["Standard (Fusha/MSA)"]["translator_de"]

    SIMILARITY_MIN = 0.88
    MIN_AR_RATIO = 0.65

    def sanitize_fixed_ar(raw_ar: str, fixed_ar: str) -> str:
        if not fixed_ar:
            return raw_ar
        fx = fixed_ar.strip()

        if re.search(r"[A-Za-z]", fx):
            return raw_ar
        if arabic_ratio(fx) < MIN_AR_RATIO:
            return raw_ar
        if similarity(raw_ar, fx) < SIMILARITY_MIN:
            return raw_ar
        return fx

    while True:
        try:
            job = input_q.get(timeout=1.0)

            if job == "STOP":
                if cloud_q is not None:
                    cloud_q.put({"id": 0, "ar": "", "final": True})
                break

            if not isinstance(job, dict):
                continue

            raw_ar = (job.get("source_ar") or "").strip()
            batch_id = int(job.get("id", 0))
            t_submitted = job.get("ts", time.time())

            log_file_ar = job.get("log_file_ar", None)
            log_file_en = job.get("log_file_en", None)
            log_file_de = job.get("log_file_de", None)

            if not raw_ar:
                continue

            # Per-job prompts
            if "prompts" in job and job["prompts"]:
                current_fixer_prompt = job["prompts"].get("fixer", current_fixer_prompt)
                current_translator_en_prompt = job["prompts"].get("translator", current_translator_en_prompt)
                current_translator_de_prompt = job["prompts"].get("translator_de", current_translator_de_prompt)

            # ✅ spellcheck BEFORE sending to any LLM (local Ollama or Groq cloud)
            raw_for_llm = raw_ar
            raw_for_llm = normalize_khutbah_staples(raw_for_llm)
            raw_for_llm = spellcheck_arabic(raw_for_llm, enable_symspell=enable_spellcheck)

            # If drift/non-arabic => do not translate, do not fix
            if arabic_ratio(raw_for_llm) < 0.50:
                corrected_ar = raw_for_llm
                final_en = "[non-arabic]"
                final_de = "[non-arabic]"

                if cloud_q is not None:
                    cloud_q.put({
                        "range": job.get("range", (batch_id, batch_id)),
                        "id": batch_id,
                        "ar": raw_for_llm,
                        "final": bool(job.get("final", False)),
                    })

                output_q.put({"type": "refined_batch", "id": batch_id, "ar_fixed": corrected_ar, "en_final": final_en, "de_final": final_de})
                continue

            # If gibberish => quarantine
            if is_gibberish_arabic(raw_for_llm):
                corrected_ar = "[غير واضح]"
                final_en = "[unclear]"
                final_de = "[unklar]"

                if cloud_q is not None:
                    cloud_q.put({
                        "range": job.get("range", (batch_id, batch_id)),
                        "id": batch_id,
                        "ar": raw_for_llm,
                        "final": bool(job.get("final", False)),
                    })

                output_q.put({"type": "refined_batch", "id": batch_id, "ar_fixed": corrected_ar, "en_final": final_en, "de_final": final_de})
                continue

            log(f"Refining Batch {batch_id}...")

            fixer_sys = (
                current_fixer_prompt.strip()
                + "\n\nCRITICAL RULES (must follow):\n"
                "- Output Arabic ONLY.\n"
                "- Do NOT paraphrase. Do NOT add new sentences. Do NOT remove sentences.\n"
                "- Only fix obvious phonetic/ASR confusions that make the sentence nonsensical.\n"
                "- If unsure, keep the original phrase unchanged.\n"
                "- Never insert any English words.\n"
            )

            fixed_candidate = robust_ollama_call({
                "model": DEFAULT_MT_MODEL,
                "messages": [
                    {"role": "system", "content": fixer_sys},
                    {"role": "user", "content": raw_for_llm},
                ],
                "stream": False,
                "options": {"temperature": 0.0, "top_p": 0.2, "num_ctx": 1600},
            }, timeout=120)

            corrected_ar = sanitize_fixed_ar(raw_for_llm, fixed_candidate)
            corrected_ar = normalize_khutbah_staples(corrected_ar)
            lat_fix = time.time() - t_submitted

            if log_file_ar:
                append_to_file(log_file_ar, f"[{get_current_ts_string()}] [Batch {batch_id}] (Lat: {lat_fix:.2f}s)\n{corrected_ar}\n")

            # Send to cloud (spellchecked raw, per your request)
            if cloud_q is not None:
                cloud_q.put({
                    "range": job.get("range", (batch_id, batch_id)),
                    "id": batch_id,
                    "ar": raw_for_llm,
                    "final": bool(job.get("final", False)),
                })

            translator_sys_en = (
                current_translator_en_prompt.strip()
                + "\n\n" + glossary_block()
                + "\n\nCRITICAL RULES (must follow):\n"
                "- Translate ONLY what is present in the Arabic input.\n"
                "- If a phrase is garbled/unclear, write [unclear] rather than guessing.\n"
                "- Output ONLY the translation.\n"
            )

            translator_sys_de = (
                current_translator_de_prompt.strip()
                + "\n\n" + glossary_block()
                + "\n\nCRITICAL RULES (must follow):\n"
                "- Translate ONLY what is present in the Arabic input.\n"
                "- If a phrase is garbled/unclear, write [unklar] rather than guessing.\n"
                "- Output ONLY the translation.\n"
            )

            final_en = robust_ollama_call({
                "model": DEFAULT_MT_MODEL,
                "messages": [
                    {"role": "system", "content": translator_sys_en},
                    {"role": "user", "content": corrected_ar},
                ],
                "stream": False,
                "options": {"temperature": 0.0, "top_p": 0.2, "num_ctx": 1800},
            }, timeout=60) or "[unclear]"

            final_de = robust_ollama_call({
                "model": DEFAULT_MT_MODEL,
                "messages": [
                    {"role": "system", "content": translator_sys_de},
                    {"role": "user", "content": corrected_ar},
                ],
                "stream": False,
                "options": {"temperature": 0.0, "top_p": 0.2, "num_ctx": 1800},
            }, timeout=60) or "[unklar]"

            lat_total = time.time() - t_submitted

            output_q.put({
                "type": "refined_batch",
                "id": batch_id,
                "ar_fixed": corrected_ar,
                "en_final": final_en,
                "de_final": final_de,
            })

            if log_file_en:
                append_to_file(log_file_en, f"[{get_current_ts_string()}] [Batch {batch_id}] (Model: {DEFAULT_MT_MODEL} | Total Lat: {lat_total:.2f}s)\n{final_en}\n")
            if log_file_de:
                append_to_file(log_file_de, f"[{get_current_ts_string()}] [Batch {batch_id}] (Model: {DEFAULT_MT_MODEL} | Total Lat: {lat_total:.2f}s)\n{final_de}\n")

            log(f"Batch {batch_id} Done (Total Lat: {lat_total:.2f}s).")

        except queue.Empty:
            continue
        except Exception as e:
            log(f"Refinement Worker Crash: {e}")
            time.sleep(0.5)

# ==========================================
# 7) AUDIO INPUT / WHISPER STREAM THREAD
# ==========================================

def get_input_devices():
    try:
        devices = sd.query_devices()
        input_devices = []
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                input_devices.append((i, d.get("name", f"Device {i}")))
        return input_devices
    except Exception as e:
        log(f"Audio Device Error: {e}")
        return []

def transcription_stream_thread(
    source,
    config,
    stop_event: threading.Event,
    refine_input_q: "queue.Queue",
    event_q: "queue.Queue",
    *,
    enable_spellcheck: bool = True,
):
    """
    Real-time mic stream:
    - WebRTC VAD (if available) -> voice-only chunks
    - rolling buffer -> faster-whisper decode
    - LocalAgreement -> stable output
    - batches -> refinement_worker
    """
    import datetime
    from collections import deque, Counter

    def get_ts():
        return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

    log_raw = config["logs"]["raw_ar"]

    def log_to_file(text):
        try:
            with open(log_raw, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception:
            pass

    _ws_re = re.compile(r"\s+")
    def norm_text(s: str) -> str:
        s = (s or "").replace("\u200f", "").replace("\u200e", "")
        s = _ws_re.sub(" ", s).strip()
        return s

    def split_words(s: str):
        s = norm_text(s)
        return s.split(" ") if s else []

    def join_words(ws):
        return " ".join([w for w in ws if w]).strip()

    def lcp_words(a, b):
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return a[:i]

    def trim_last_word(words):
        return words[:-1] if words else words

    # Rolling audio buffer
    class RollingAudioBuffer:
        def __init__(self, sr: int, max_sec: float):
            self.sr = int(sr)
            self.max_samples = int(max_sec * self.sr)
            self._chunks = deque()
            self.n_samples = 0

        def append(self, audio: np.ndarray) -> None:
            if audio is None or len(audio) == 0:
                return
            a = np.asarray(audio, dtype=np.float32).reshape(-1)
            self._chunks.append(a)
            self.n_samples += len(a)
            self._trim_to_max()

        def _trim_to_max(self) -> None:
            while self.n_samples > self.max_samples and self._chunks:
                extra = self.n_samples - self.max_samples
                left = self._chunks[0]
                if len(left) <= extra:
                    self._chunks.popleft()
                    self.n_samples -= len(left)
                else:
                    self._chunks[0] = left[extra:]
                    self.n_samples -= extra

        def get_buffer(self) -> np.ndarray:
            if not self._chunks:
                return np.zeros((0,), dtype=np.float32)
            return np.concatenate(self._chunks)

        def clear(self):
            self._chunks.clear()
            self.n_samples = 0

    # VAD
    try:
        import webrtcvad
        _VAD_OK = True
    except Exception:
        _VAD_OK = False

    class VacIterator:
        def __init__(
            self,
            sr=16000,
            frame_sec=0.02,
            min_silence_ms=500,
            pad_ms=100,
            min_voiced_sec=1.0,
            vad_mode=2,
        ):
            self.sr = int(sr)
            self.frame_sec = float(frame_sec)
            self.frame_samples = int(self.sr * self.frame_sec)
            self.min_silence_ms = int(min_silence_ms)
            self.pad_samples = int(self.sr * (pad_ms / 1000.0))
            self.min_voiced_samples = int(self.sr * float(min_voiced_sec))

            self.vad = webrtcvad.Vad(vad_mode) if _VAD_OK else None
            self.in_voice = False
            self.silence_ms = 0

            self.pad_buf = deque()
            self.voiced_chunks = []
            self.voiced_samples = 0

        def _is_speech(self, frame_f32: np.ndarray) -> bool:
            if self.vad is None:
                return float(np.sqrt(np.mean(frame_f32 * frame_f32))) > 0.01
            pcm16 = np.clip(frame_f32 * 32768.0, -32768, 32767).astype(np.int16).tobytes()
            return self.vad.is_speech(pcm16, self.sr)

        def push_frame(self, frame_f32: np.ndarray):
            out = []
            if frame_f32 is None or len(frame_f32) == 0:
                return out

            if len(frame_f32) != self.frame_samples:
                if len(frame_f32) < self.frame_samples:
                    frame_f32 = np.pad(frame_f32, (0, self.frame_samples - len(frame_f32)))
                else:
                    frame_f32 = frame_f32[:self.frame_samples]

            self.pad_buf.append(frame_f32)
            total = sum(len(x) for x in self.pad_buf)
            while total > self.pad_samples and self.pad_buf:
                drop = total - self.pad_samples
                x = self.pad_buf[0]
                if len(x) <= drop:
                    self.pad_buf.popleft()
                    total -= len(x)
                else:
                    self.pad_buf[0] = x[drop:]
                    total -= drop

            speech = self._is_speech(frame_f32)

            if speech:
                if not self.in_voice:
                    self.in_voice = True
                    if self.pad_buf:
                        pad = np.concatenate(list(self.pad_buf))
                        self.voiced_chunks.append(pad)
                        self.voiced_samples += len(pad)
                    self.silence_ms = 0

                self.voiced_chunks.append(frame_f32)
                self.voiced_samples += len(frame_f32)
                self.silence_ms = 0

                if self.voiced_samples >= self.min_voiced_samples:
                    chunk = np.concatenate(self.voiced_chunks) if self.voiced_chunks else np.zeros((0,), np.float32)
                    if len(chunk) > 0:
                        out.append((chunk, False))
                    self.voiced_chunks = []
                    self.voiced_samples = 0
                return out

            if self.in_voice:
                self.silence_ms += int(self.frame_sec * 1000)
                if self.silence_ms >= self.min_silence_ms:
                    chunk = np.concatenate(self.voiced_chunks) if self.voiced_chunks else np.zeros((0,), np.float32)
                    if len(chunk) > 0:
                        out.append((chunk, True))
                    self.in_voice = False
                    self.silence_ms = 0
                    self.voiced_chunks = []
                    self.voiced_samples = 0

            return out

    # Load Whisper model for this session (cached, but unloadable on STOP)
    try:
        model = load_whisper_model_cached(config["model_size"], config["device"], config["compute_type"])
    except Exception as e:
        event_q.put(("error", f"Whisper load failed: {e}"))
        return

    # Audio source
    is_live_mic = isinstance(source, int)
    sr = 16000
    stream = None
    wf = None

    if is_live_mic:
        try:
            stream = sd.InputStream(
                device=source,
                channels=1,
                samplerate=sr,
                dtype="float32",
                blocksize=int(sr * 0.02),
            )
            stream.start()
        except Exception as e:
            event_q.put(("error", f"Mic Error: {e}"))
            return
    else:
        try:
            wf = wave.open(source, "rb")
            sr = wf.getframerate()
        except Exception as e:
            event_q.put(("error", f"File Error: {e}"))
            return

    # Filters/tuning
    hallucinations_re = re.compile(
        r"(Subtitle|Translated|Amara|MBC|Copyright|Rights|Reserved|Music|"
        r"نانسي|ترجمة|اشتركوا|اشترك|الحقوق|حقوق|محفوظة|قناة|تابعونا|"
        r"شكرا|شكراً)",
        re.IGNORECASE,
    )

    VAD_FRAME_SEC = 0.02
    MIN_NONVOICE_MS = 500
    VOICE_PAD_MS = 100
    MIN_VOICED_ACCUM_SEC = float(config.get("min_voiced_accum_sec", 1.0))

    MAX_WINDOW_SEC = float(config.get("max_window_sec", 15.0))
    AUDIO_MIN_SEC = float(config.get("audio_min_sec", 1.0))

    audio_buf = RollingAudioBuffer(sr=sr, max_sec=MAX_WINDOW_SEC)
    vac = VacIterator(
        sr=sr,
        frame_sec=VAD_FRAME_SEC,
        min_silence_ms=MIN_NONVOICE_MS,
        pad_ms=VOICE_PAD_MS,
        min_voiced_sec=MIN_VOICED_ACCUM_SEC,
        vad_mode=int(config.get("vad_mode", 2)),
    )

    committed_words = []
    prev_hyp_words = []
    chunk_counter = 0
    accumulated_text = ""
    last_send_time = time.time()

    refine_every = int(config.get("refine_every", 4))
    batch_buf = []
    batch_start_id = None
    batch_end_id = None

    def flush_refine_batch(force_final: bool = False):
        nonlocal batch_buf, batch_start_id, batch_end_id
        if not batch_buf:
            return

        batch_text = "\n\n".join(batch_buf).strip()
        if not batch_text:
            batch_buf = []
            batch_start_id = None
            batch_end_id = None
            return

        job = {
            "id": batch_end_id or batch_start_id or 0,
            "range": (batch_start_id or 0, batch_end_id or batch_start_id or 0),
            "source_ar": batch_text,
            "ts": time.time(),
            "log_file_ar": config["logs"].get("fixed_ar"),
            "log_file_en": config["logs"].get("final_en"),
            "log_file_de": config["logs"].get("final_de"),
            "prompts": config.get("prompts"),
            "final": bool(force_final),
        }
        refine_input_q.put(job)

        batch_buf = []
        batch_start_id = None
        batch_end_id = None

    def too_repetitive(ws):
        if len(ws) < 12:
            return False
        c = Counter(ws)
        top = c.most_common(1)[0][1]
        return top / len(ws) > 0.35

    try:
        while not stop_event.is_set():
            # Read one frame
            if is_live_mic:
                if not stream or not stream.active:
                    break
                data, _overflow = stream.read(int(sr * VAD_FRAME_SEC))
                frame = data.flatten().astype(np.float32)
            else:
                raw_bytes = wf.readframes(int(sr * VAD_FRAME_SEC))
                if not raw_bytes:
                    break
                frame = np.frombuffer(raw_bytes, np.int16).astype(np.float32) / 32768.0

            if len(frame) == 0:
                break

            t_capture_end = time.time()
            yielded = vac.push_frame(frame)
            if not yielded:
                continue

            for voiced_chunk, is_final in yielded:
                if voiced_chunk is None or len(voiced_chunk) == 0:
                    continue

                audio_buf.append(voiced_chunk)

                if audio_buf.n_samples < int(AUDIO_MIN_SEC * sr):
                    continue

                window_audio = audio_buf.get_buffer()

                # Decode
                t_infer_start = time.time()
                transcribe_kwargs = dict(
                    beam_size=int(config.get("beam_size", 2)),
                    language=config.get("language", "ar"),
                    condition_on_previous_text=False,
                    word_timestamps=False,
                    no_speech_threshold=0.5,
                    log_prob_threshold=-1.0,
                    compression_ratio_threshold=2.1,
                    temperature=0.0,
                    repetition_penalty=1.05,
                )

                try:
                    segments_gen, _info = model.transcribe(window_audio, **transcribe_kwargs)
                    segments = list(segments_gen)
                except Exception as e:
                    event_q.put(("error", f"Whisper Error: {e}"))
                    continue

                t_infer_end = time.time()
                infer_dur = t_infer_end - t_infer_start
                lag_behind = t_infer_end - t_capture_end

                hyp_text = norm_text("".join([getattr(s, "text", "") for s in segments]))
                m = hallucinations_re.search(hyp_text)
                if m:
                    hyp_text = hyp_text[:m.start()].strip()
                    if not hyp_text:
                        continue

                hyp_words = split_words(hyp_text)

                hyp_words_safe = trim_last_word(hyp_words) if not is_final else hyp_words
                hyp_words_safe = [w for w in hyp_words_safe if not hallucinations_re.search(w)]

                if too_repetitive(hyp_words_safe):
                    audio_buf.clear()
                    prev_hyp_words = []
                    committed_words = []
                    accumulated_text = ""
                    continue

                stable = lcp_words(prev_hyp_words, hyp_words_safe) if prev_hyp_words else []
                prev_hyp_words = hyp_words_safe

                if len(stable) > len(committed_words):
                    new_words = stable[len(committed_words):]
                    if committed_words and new_words and new_words[0] == committed_words[-1]:
                        new_words = new_words[1:]
                    if new_words:
                        committed_words.extend(new_words)
                        text_chunk = join_words(new_words)
                        accumulated_text = (accumulated_text + " " + text_chunk).strip()

                if is_final and len(hyp_words_safe) > len(committed_words):
                    tail = hyp_words_safe[len(committed_words):]
                    committed_words = hyp_words_safe[:]
                    tail_txt = join_words(tail)
                    if tail_txt:
                        accumulated_text = (accumulated_text + " " + tail_txt).strip()

                # Emit
                if accumulated_text and (len(accumulated_text) > 260 or (time.time() - last_send_time > 3.0) or is_final):
                    chunk_counter += 1
                    clean_raw = accumulated_text.strip()

                    log_msg = f"[{get_ts()}] [{chunk_counter}] (Lag: {lag_behind:.2f}s | Infer: {infer_dur:.2f}s) {clean_raw}"
                    print(log_msg)
                    log_to_file(log_msg)

                    # Column 1 shows RAW
                    event_q.put(("update", {"id": chunk_counter, "ar": clean_raw}))

                    # ✅ Spellcheck BEFORE sending to LLM pipeline
                    clean_for_llm = normalize_khutbah_staples(clean_raw)
                    clean_for_llm = spellcheck_arabic(clean_for_llm, enable_symspell=enable_spellcheck)

                    if batch_start_id is None:
                        batch_start_id = chunk_counter
                    batch_end_id = chunk_counter
                    batch_buf.append(clean_for_llm)

                    if len(batch_buf) >= refine_every:
                        flush_refine_batch(force_final=is_final)

                    accumulated_text = ""
                    last_send_time = time.time()

                if is_final:
                    audio_buf.clear()
                    prev_hyp_words = []
                    committed_words = []
    finally:
        # Flush remaining batch on exit
        flush_refine_batch(force_final=True)
        event_q.put(("status", "stream_finished"))

        try:
            if wf:
                wf.close()
        except Exception:
            pass
        try:
            if stream:
                stream.stop()
                stream.close()
        except Exception:
            pass

# ==========================================
# 8) UI + SESSION CONTROL
# ==========================================

def _init_app_state():
    if "app_booted" in st.session_state:
        return

    st.session_state.app_booted = True
    st.session_state.streaming = False

    # Current session fields (reset on each Start)
    st.session_state.uid = ""
    st.session_state.chunks = []
    st.session_state.refined_blocks_ar = []
    st.session_state.refined_blocks_en = []
    st.session_state.refined_blocks_de = []
    st.session_state.cloud_blocks = []
    st.session_state.last_chunk_id = 0

    # Thread controls
    st.session_state.stop_event = threading.Event()
    st.session_state.stream_thread = None

    # Worker queues (persist across sessions)
    st.session_state.refine_in = queue.Queue()
    st.session_state.refine_out = queue.Queue()
    st.session_state.event_q = queue.Queue()
    st.session_state.cloud_in = queue.Queue()
    st.session_state.cloud_out = queue.Queue()

    # Worker threads started once
    st.session_state.refinement_thread_started = False
    st.session_state.cloud_thread_started = False

def start_new_session():
    # New UID + clear UI blocks + new logs
    st.session_state.uid = str(uuid.uuid4())[:8]
    st.session_state.chunks = []
    st.session_state.refined_blocks_ar = []
    st.session_state.refined_blocks_en = []
    st.session_state.refined_blocks_de = []
    st.session_state.cloud_blocks = []
    st.session_state.last_chunk_id = 0

def stop_stream_now():
    # Stop mic thread
    try:
        st.session_state.stop_event.set()
    except Exception:
        pass

    # Try join quickly (avoid blocking too long in Streamlit)
    t = st.session_state.stream_thread
    if t and isinstance(t, threading.Thread) and t.is_alive():
        try:
            t.join(timeout=1.5)
        except Exception:
            pass

    st.session_state.streaming = False

    # Flush cloud buffer
    try:
        st.session_state.cloud_in.put({"id": st.session_state.last_chunk_id, "ar": "", "final": True})
    except Exception:
        pass

    # ✅ GPU deload (best-effort)
    unload_whisper_model_best_effort()

def main():
    st.set_page_config(layout="wide", page_title="Khutbah AI")
    _init_app_state()

    # Start background workers once
    if not st.session_state.refinement_thread_started:
        t_ref = threading.Thread(
            target=refinement_worker,
            args=(st.session_state.refine_in, st.session_state.refine_out, st.session_state.cloud_in),
            kwargs={"enable_spellcheck": True},  # will be overridden by sidebar toggle via session_state
            daemon=True,
        )
        add_script_run_ctx(t_ref)
        t_ref.start()
        st.session_state.refinement_thread_started = True

    if not st.session_state.cloud_thread_started:
        t_cloud = threading.Thread(
            target=cloud_polish_worker,
            args=(st.session_state.cloud_in, st.session_state.cloud_out),
            kwargs={"flush_every": 8, "session_uid": "persistent", "enable_spellcheck": True},
            daemon=True,
        )
        add_script_run_ctx(t_cloud)
        t_cloud.start()
        st.session_state.cloud_thread_started = True

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("🗣️ Dialect / Style")
        styles = list(PROMPT_TEMPLATES.keys())
        selected_style = st.selectbox("Select Speech Style", styles, index=styles.index("Standard (Fusha/MSA)") if "Standard (Fusha/MSA)" in styles else 0)
        st.info(f"Mode: **{selected_style}**")
        st.divider()

        # Spellcheck controls
        st.subheader("🔎 Arabic Spellcheck")
        enable_spell = st.checkbox("Enable spellcheck before LLMs", value=True)
        st.caption("Uses safe khutbah fixes + (optional) SymSpell if AR_SYMSPELL_DICT is provided.")

        st.divider()

        input_mode = st.radio("Source", ["Microphone", "File Upload"], index=0)
        mic_index = None
        if input_mode == "Microphone":
            input_devices = get_input_devices()
            if input_devices:
                opts = [d[1] for d in input_devices]
                sel = st.selectbox("Device", opts)
                for idx, name in input_devices:
                    if name == sel:
                        mic_index = idx
                        break
            else:
                st.warning("No input devices detected.")

        st.subheader("🚀 Hardware Optimization")
        model_size = st.selectbox("Whisper Size", ["distil-large-v3", "large-v3", "medium"], index=1)
        compute_type = st.selectbox("Compute Type", ["float16", "int8_float16", "int8"], index=2)
        device = st.radio("Compute Device", ["cuda", "cpu"], index=0)

        refine_every = st.slider("Refine Batch Size", 2, 10, 3)

        st.divider()
        if st.button("🧹 RESET (Stop + New Session)", use_container_width=True):
            stop_stream_now()
            start_new_session()
            st.rerun()

    st.title("🕌 Khutbah AI: Real-time Transcription")
    base_logs_path = os.path.join(os.getcwd(), "logs")
    st.caption(f"📂 **LOG FILES SAVING TO:** `{base_logs_path}`")

    # File upload mode (kept simple; mic is your priority)
    source_to_pass = None
    if input_mode == "File Upload":
        u = st.file_uploader("Upload Audio/Video", type=["mp4", "wav"])
        if u:
            with tempfile.NamedTemporaryFile(delete=False) as t:
                t.write(u.read())
                vp = t.name
            wp = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4().hex[:8]}.wav")

            with st.status("Extracting Audio..."):
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", vp, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wp],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=True,
                    )
                    st.success("Extraction complete")
                except Exception as e:
                    st.error(f"FFMPEG Error: {e}")
                    st.stop()

            source_to_pass = wp
    else:
        if mic_index is not None:
            source_to_pass = mic_index

    # Layout
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.subheader("1. Raw Arabic")
    with c2: st.subheader("2. Refined Arabic")
    with c3: st.subheader("3. Refined English")
    with c4: st.subheader("4. Refined German")
    with c5: st.subheader("5. Cloud Translation (EN/DE)")

    box_raw = c1.empty()
    box_fixed = c2.empty()
    box_final_en = c3.empty()
    box_final_de = c4.empty()
    box_cloud = c5.empty()

    # Controls
    if source_to_pass is not None:
        if not st.session_state.streaming:
            if st.button("▶️ START STREAM", type="primary", use_container_width=True):
                # New session + new logs for each start
                start_new_session()

                st.session_state.streaming = True
                st.session_state.stop_event = threading.Event()

                mt_model_safe = sanitize_filename(DEFAULT_MT_MODEL)

                config = {
                    "model_size": model_size,
                    "device": device,
                    "compute_type": compute_type,
                    "refine_every": refine_every,
                    "max_window_sec": 25.0,
                    "beam_size": 2,
                    "prompts": PROMPT_TEMPLATES[selected_style],
                    "logs": {
                        "raw_ar": get_log_path(f"log_{st.session_state.uid}_1_raw_ar.txt"),
                        "fixed_ar": get_log_path(f"log_{st.session_state.uid}_2_fixed_ar.txt"),
                        "final_en": get_log_path(f"log_{st.session_state.uid}_{mt_model_safe}_en.txt"),
                        "final_de": get_log_path(f"log_{st.session_state.uid}_{mt_model_safe}_de.txt"),
                    }
                }

                t_stream = threading.Thread(
                    target=transcription_stream_thread,
                    args=(source_to_pass, config, st.session_state.stop_event, st.session_state.refine_in, st.session_state.event_q),
                    kwargs={"enable_spellcheck": enable_spell},
                    daemon=True,
                )
                add_script_run_ctx(t_stream)
                t_stream.start()
                st.session_state.stream_thread = t_stream

                st.rerun()
        else:
            if st.button("⏹️ STOP STREAM", type="secondary", use_container_width=True):
                stop_stream_now()
                st.rerun()

    # Static display when not streaming
    if not st.session_state.streaming:
        box_raw.text_area("Raw", value="\n\n".join([c["ar"] for c in st.session_state.chunks]), height=600, key="static_raw")
        box_fixed.text_area("Refined Arabic", value="\n\n".join(st.session_state.refined_blocks_ar), height=600, key="static_fixed")
        box_final_en.text_area("Refined English", value="\n\n".join(st.session_state.refined_blocks_en), height=600, key="static_final_en")
        box_final_de.text_area("Refined German", value="\n\n".join(st.session_state.refined_blocks_de), height=600, key="static_final_de")
        box_cloud.text_area("Cloud Translation", value="\n\n".join(st.session_state.cloud_blocks), height=600, key="static_cloud")
        return

    # Streaming UI loop (lightweight polling)
    # NOTE: Streamlit reruns frequently; keep this loop short + cooperative.
    t0 = time.time()
    while st.session_state.streaming and not st.session_state.stop_event.is_set():
        has_data = False

        # Drain event_q (raw chunks)
        try:
            while True:
                t, p = st.session_state.event_q.get_nowait()
                if t == "update":
                    st.session_state.chunks.append(p)
                    st.session_state.last_chunk_id = p.get("id", st.session_state.last_chunk_id)
                    has_data = True
                elif t == "error":
                    st.error(f"Error: {p}")
                    stop_stream_now()
                    has_data = True
                    break
                elif t == "status" and p == "stream_finished":
                    stop_stream_now()
                    has_data = True
                    break
        except queue.Empty:
            pass

        # Drain refine_out (local refined)
        try:
            while True:
                p = st.session_state.refine_out.get_nowait()
                if p.get("type") == "refined_batch":
                    st.session_state.refined_blocks_ar.append(f"[{p['id']}] {p['ar_fixed']}")
                    st.session_state.refined_blocks_en.append(f"[{p['id']}] {p['en_final']}")
                    st.session_state.refined_blocks_de.append(f"[{p['id']}] {p['de_final']}")
                    has_data = True
        except queue.Empty:
            pass

        # Drain cloud_out (cloud refined)
        try:
            while True:
                p = st.session_state.cloud_out.get_nowait()
                st.session_state.cloud_blocks.append(
                    f"[{p['range'][0]}–{p['range'][1]}] ({p['lang']}, {p.get('model','cloud')})\n{p['text']}"
                )
                has_data = True
        except queue.Empty:
            pass

        if has_data:
            iter_id = str(uuid.uuid4())[:8]
            box_raw.text_area("Raw", value="\n\n".join([c["ar"] for c in st.session_state.chunks]), height=600, key=f"raw_{iter_id}")
            box_fixed.text_area("Refined Arabic", value="\n\n".join(st.session_state.refined_blocks_ar), height=600, key=f"fixed_{iter_id}")
            box_final_en.text_area("Refined English", value="\n\n".join(st.session_state.refined_blocks_en), height=600, key=f"final_en_{iter_id}")
            box_final_de.text_area("Refined German", value="\n\n".join(st.session_state.refined_blocks_de), height=600, key=f"final_de_{iter_id}")
            box_cloud.text_area("Cloud Translation", value="\n\n".join(st.session_state.cloud_blocks), height=600, key=f"cloud_{iter_id}")

        # keep loop responsive and short
        time.sleep(0.15)

        # safety: avoid long blocking in a single run
        if time.time() - t0 > 6.0:
            st.rerun()

if __name__ == "__main__":
    main()
