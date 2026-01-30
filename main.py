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
from faster_whisper import WhisperModel
from streamlit.runtime.scriptrunner import add_script_run_ctx
from typing import List, Optional, Dict, Any

from groq import Groq
import groq
import torch
from telegram_sink import telegram_sink_worker

# ==========================================
# 1. CONFIGURATION & PROMPTS
# ==========================================

OLLAMA_API_URL = "https://j4clsjgwgcvkzg-4000.proxy.runpod.net/llm/chat"
LLM_API_HEADERS = {
    "x-api-key": "c129c38d4d0b559c2b6f823167ad7f7ee3cbcccf941103ae57612ccd5457a817",
    "Content-Type": "application/json",
}
DEFAULT_MT_MODEL = "command-r:35b-8k"

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
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

GROQ_MODEL_FALLBACK: List[str] = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k2-instruct",
    "",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "groq/compound-mini",
]



GLOSSARY = {
    "الله": "Allah",
    "القرآن": "the Qur'an",
    "محمد": "Muhammad",
    "صلاة": "Salah",
    "زكاة": "Zakah",
    "جهاد": "Jihad",
    "تقوى": "Taqwa"
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_current_ts_string():
    now = datetime.datetime.now()
    return now.strftime("%H:%M:%S.%f")[:-3]

def log(msg):
    print(f"[{get_current_ts_string()}] {msg}", flush=True)

def get_log_path(filename):
    base_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, filename)


def sanitize_filename(name: str) -> str:
    """Make a safe ascii-ish filename segment from a model or label."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "")
    cleaned = cleaned.strip("._")
    return cleaned or "unnamed"

def append_to_file(filepath, text):
    if not text: return
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception as e:
        log(f"FILE ERROR ({filepath}): {e}")

def get_input_devices():
    try:
        devices = sd.query_devices()
        input_devices = []
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                input_devices.append((i, d['name']))
        return input_devices
    except Exception as e:
        log(f"Audio Device Error: {e}")
        return []
    

_AR_RE = re.compile(r'[\u0600-\u06FF]')

def arabic_ratio(text: str) -> float:
    """Rough ratio of Arabic letters among alphabetic chars."""
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    ar_letters = sum(1 for c in letters if _AR_RE.match(c))
    return ar_letters / max(1, len(letters))

def is_good_arabic_segment(text: str, min_len: int = 20) -> bool:
    """
    Heuristic filter to prevent drift:
    - Must be mostly Arabic.
    - Must not contain common boilerplate hallucinations.
    - Must be long enough (tunable).
    """
    if not text:
        return False
    t = text.strip()
    if len(t) < min_len:
        return False

    # Common boilerplate / subtitle hallucinations (Arabic + English)
    if re.search(r'(نانسي|ترجمة|اشتركوا|اشترك|تابعونا|حقوق|محفوظة|موسيقى|قناة|Subtitle|Translated|Amara|MBC|Copyright|Rights|Reserved|Music|Nancy|Nana)',
                 t, re.IGNORECASE):
        return False

    return arabic_ratio(t) >= 0.75

def glossary_block() -> str:
    """Embed glossary as instructions (avoids fragile placeholders)."""
    lines = [f"- {ar} => {en}" for ar, en in GLOSSARY.items()]
    return "GLOSSARY (must follow exactly):\n" + "\n".join(lines)

def similarity(a: str, b: str) -> float:
    """Useful if you decide to guard the fixer against rewrites."""
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


def gec_correct_ar(text: str, tok, model, max_new_tokens: int = 256) -> str:
    text = (text or "").strip()
    if not text:
        return text

    device = next(model.parameters()).device
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            do_sample=False,
            length_penalty=1.0,
            early_stopping=True,
        )
    return tok.decode(out_ids[0], skip_special_tokens=True).strip()


def robust_ollama_call(payload, timeout=30, retries=1):
    """Call the /llm/chat endpoint with a messages array."""
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

            # --- CLEANER ---
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text = re.sub(r'^(Here is|Sure|Translation|Corrected|Okay|I have translated).*?:', '', text, flags=re.IGNORECASE).strip()
            text = re.sub(r'(Note:|P.S.|Analysis:).*', '', text, flags=re.IGNORECASE).strip()
            text = text.strip('"').strip("'")
            # ---------------

            if re.search(r'[\u4e00-\u9fff]', text):
                return None

            return text
        except Exception as e:
            log(f"[API] Error: {e}")
        if attempt < retries:
            time.sleep(1)
    return None


def parse_tagged_response(text: str) -> Dict[str, str]:
    """Parse [Arabic] / [German] / [English] tagged response from command-r."""
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

def protect_terms(text):
    mapping = {}
    out = text
    for i, (ar, en) in enumerate(GLOSSARY.items()):
        if ar in out:
            key = f"§§TERM{i}§§"
            mapping[key] = en
            out = out.replace(ar, key)
    return out, mapping

def restore_terms(text, mapping):
    out = text
    for key, val in mapping.items():
        out = out.replace(key, val)
    return out


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


def cloud_polish_worker(
    cloud_in_q: "queue.Queue",
    cloud_out_q: "queue.Queue",
    *,
    target_langs: Optional[List[str]] = None,
    flush_every: int = 8,
    session_uid: str = "",
):
    if not groq_client:
        log("[CLOUD] GROQ_API_KEY not set => cloud disabled.")
        # Drain queue so it doesn't grow forever
        while True:
            x = cloud_in_q.get()
            if x == "STOP":
                return

    buf = []
    start_id = None
    end_id = None
    last_flush = time.time()

    # Sliding window of previous results for context
    cloud_context_window: List[Dict[str, str]] = []

    def make_system_prompt() -> str:
        return (
            "You are an Arabic Islamic sermon (khutba) editor and translator.\n\n"
            "STEPS:\n"
            "1. Fix spelling and grammar errors in the Arabic ASR transcript. "
            "Keep the original wording and style. Do not paraphrase or modernize classical expressions.\n"
            "2. Translate the corrected Arabic into German.\n"
            "3. Translate the corrected Arabic into English.\n\n"
            "OUTPUT FORMAT (follow exactly):\n\n"
            "[Arabic]\n<corrected Arabic text here>\n\n"
            "[German]\n<German translation here>\n\n"
            "[English]\n<English translation here>\n\n"
            "RULES:\n"
            "- Preserve all Quran verses and hadith exactly as spoken.\n"
            "- Keep Islamic terms like صلى الله عليه وسلم, سبحانه وتعالى, إن شاء الله "
            "untranslated in parentheses.\n"
            "- Do not add commentary or explanation.\n"
            "- Preserve paragraph breaks from the original.\n"
            "- If a phrase is garbled/unclear, write [unclear] in English or [unklar] in German.\n"
            "- If the input is not Arabic, output [non-arabic] in all three sections.\n"
            "- You may receive previous context (already processed) above the target text. "
            "Use it ONLY for understanding continuity. Do NOT include the context in your output. "
            "Output ONLY the fix and translations of the TARGET section."
        )

    def build_cloud_user_message(batch_text: str) -> str:
        parts = []
        if cloud_context_window:
            ctx_lines = [entry.get("fixed_ar", "") for entry in cloud_context_window]
            parts.append("### PREVIOUS CONTEXT (read-only, do NOT include in output):\n" + "\n".join(ctx_lines))
        parts.append("### TARGET TEXT (fix and translate this):\n" + batch_text)
        return "\n\n".join(parts)

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
        batch = "\n\n".join(buf)

        if groq_client:
            log(f"[CLOUD] Flushing {start_id}-{end_id} ({len(buf)} parts) -> langs=['en','de'] with Arabic fix")
            user_content = build_cloud_user_message(batch)
            messages = [
                {"role": "system", "content": make_system_prompt()},
                {"role": "user", "content": user_content},
            ]

            _cloud_t0 = time.time()
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
            _cloud_lat = time.time() - _cloud_t0

            if result and result.get("text"):
                raw_resp = result["text"].strip()

                parsed = parse_tagged_response(raw_resp)
                fixed_ar = parsed.get("fixed_ar", "").strip()
                en_text = parsed.get("en", "").strip()
                de_text = parsed.get("de", "").strip()

                safe_model = sanitize_filename(result.get("model", "cloud"))

                if fixed_ar:
                    log_name_ar = f"log_{session_uid}_cloud_{safe_model}_ar.txt" if session_uid else f"log_cloud_{safe_model}_ar.txt"
                    append_to_file(
                        get_log_path(log_name_ar),
                        f"[{get_current_ts_string()}] [{start_id}-{end_id}] (ar-fixed, {result['model']})\n{fixed_ar}\n"
                    )
                    cloud_out_q.put({
                        "range": (start_id, end_id),
                        "model": result["model"],
                        "text": fixed_ar,
                        "lang": "ar-fixed",
                        "lat_s": _cloud_lat,
                    })

                if en_text:
                    log_name_en = f"log_{session_uid}_cloud_{safe_model}_en.txt" if session_uid else f"log_cloud_{safe_model}_en.txt"
                    append_to_file(
                        get_log_path(log_name_en),
                        f"[{get_current_ts_string()}] [{start_id}-{end_id}] (en, {result['model']})\n{en_text}\n"
                    )
                    cloud_out_q.put({
                        "range": (start_id, end_id),
                        "model": result["model"],
                        "text": en_text,
                        "lang": "en",
                        "lat_s": _cloud_lat,
                    })

                if de_text:
                    log_name_de = f"log_{session_uid}_cloud_{safe_model}_de.txt" if session_uid else f"log_cloud_{safe_model}_de.txt"
                    append_to_file(
                        get_log_path(log_name_de),
                        f"[{get_current_ts_string()}] [{start_id}-{end_id}] (de, {result['model']})\n{de_text}\n"
                    )
                    cloud_out_q.put({
                        "range": (start_id, end_id),
                        "model": result["model"],
                        "text": de_text,
                        "lang": "de",
                        "lat_s": _cloud_lat,
                    })

                # Update sliding window context
                cloud_context_window.append({
                    "fixed_ar": fixed_ar or batch,
                    "en": en_text,
                    "de": de_text,
                })
                if len(cloud_context_window) > CONTEXT_WINDOW_SIZE:
                    cloud_context_window.pop(0)

        buf = []
        start_id = None
        end_id = None
        last_flush = time.time()

# ==========================================
# 3. WORKERS
# ==========================================

@st.cache_resource(max_entries=1)
def load_whisper_model(model_size, device, compute_type):
    id_map = {
        "distil-large-v3": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "large-v3": "Systran/faster-whisper-large-v3",
        "medium": "Systran/faster-whisper-medium"
    }
    hf_id = id_map.get(model_size, model_size)
    log(f"Loading Whisper: {hf_id} (Device: {device}, Type: {compute_type})...")
    try:
        gc.collect()
        model = WhisperModel(hf_id, device=device, compute_type=compute_type)
        log("Whisper Loaded Successfully.")
        return model
    except Exception as e:
        log(f"CRITICAL: Whisper Load Failed: {e}")
        return None


# --- WORKER 1: REFINEMENT ---
CONTEXT_WINDOW_SIZE = 15  # max previous chunks kept for context

def refinement_worker(input_q, output_q, cloud_q=None, config=None):
    log("Refinement Worker: STARTING")

    # Sliding window of previous results for context
    context_window: List[Dict[str, str]] = []  # each: {"raw": ..., "fixed_ar": ..., "en": ..., "de": ...}

    def make_ollama_system_prompt() -> str:
        return (
            "You are an Arabic Islamic sermon (khutba) editor and translator.\n\n"
            "STEPS:\n"
            "1. Fix spelling and grammar errors in the Arabic ASR transcript. "
            "Keep the original wording and style. Do not paraphrase or modernize classical expressions.\n"
            "2. Translate the corrected Arabic into German.\n"
            "3. Translate the corrected Arabic into English.\n\n"
            "OUTPUT FORMAT (follow exactly):\n\n"
            "[Arabic]\n<corrected Arabic text here>\n\n"
            "[German]\n<German translation here>\n\n"
            "[English]\n<English translation here>\n\n"
            "RULES:\n"
            "- Preserve all Quran verses and hadith exactly as spoken.\n"
            "- Keep Islamic terms like صلى الله عليه وسلم, سبحانه وتعالى, إن شاء الله "
            "untranslated in parentheses.\n"
            "- Do not add commentary or explanation.\n"
            "- Preserve paragraph breaks from the original.\n"
            "- If a phrase is garbled/unclear, write [unclear] in English or [unklar] in German.\n"
            "- If the input is not Arabic, output [non-arabic] in all three sections.\n"
            "- You may receive previous context (already processed) above the target text. "
            "Use it ONLY for understanding continuity. Do NOT include the context in your output. "
            "Output ONLY the fix and translations of the TARGET section.\n\n"
            + glossary_block()
        )

    def build_user_message(raw_ar: str) -> str:
        """Build user message with sliding window context + target text."""
        parts = []
        if context_window:
            ctx_lines = []
            for entry in context_window:
                ctx_lines.append(entry.get("fixed_ar", entry.get("raw", "")))
            parts.append("### PREVIOUS CONTEXT (read-only, do NOT include in output):\n" + "\n".join(ctx_lines))
        parts.append("### TARGET TEXT (fix and translate this):\n" + raw_ar)
        return "\n\n".join(parts)

    while True:
        try:
            job = input_q.get(timeout=2)
            if job is None:
                continue
            if job == "STOP":
                if cloud_q is not None:
                    cloud_q.put({"id": 0, "ar": "", "final": True})
                break

            raw_ar = (job.get("source_ar") or "").strip()
            batch_id = job.get("id", 0)
            log_file_ar = job.get("log_file_ar", None)
            log_file_en = job.get("log_file_en", None)
            log_file_de = job.get("log_file_de", None)
            t_submitted = job.get("ts", time.time())

            if not raw_ar:
                continue

            # If the speaker switched language (not Arabic), skip
            if arabic_ratio(raw_ar) < 0.50:
                corrected_ar = raw_ar
                final_en = "[non-arabic]"
                final_de = "[non-arabic]"
                if cloud_q is not None:
                    cloud_q.put({
                        "range": job.get("range", (batch_id, batch_id)),
                        "id": batch_id,
                        "ar": raw_ar,
                        "final": bool(job.get("final", False)),
                    })
                output_q.put({
                    "type": "refined_batch",
                    "id": batch_id,
                    "ar_fixed": corrected_ar,
                    "en_final": final_en,
                    "de_final": final_de,
                    "lat_s": 0.0,
                })
                continue

            log(f"Refining Batch {batch_id}...")

            # -------------------------
            # Single unified fix + translate call
            # -------------------------
            user_msg = build_user_message(raw_ar)

            result_text = robust_ollama_call({
                "model": DEFAULT_MT_MODEL,
                "messages": [
                    {"role": "system", "content": make_ollama_system_prompt()},
                    {"role": "user", "content": user_msg}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_ctx": 4096
                }
            }, timeout=120)

            # Parse tagged [Arabic]/[German]/[English] response
            parsed = parse_tagged_response(result_text) if result_text else {}
            corrected_ar = parsed.get("fixed_ar", "").strip()
            final_en = parsed.get("en", "").strip()
            final_de = parsed.get("de", "").strip()

            # Fallback if JSON parse failed but we got some text
            if not corrected_ar:
                corrected_ar = raw_ar
            if not final_en:
                final_en = "[unclear]"
            if not final_de:
                final_de = "[unklar]"

            lat_total = time.time() - t_submitted

            # Update sliding window context
            context_window.append({
                "raw": raw_ar,
                "fixed_ar": corrected_ar,
                "en": final_en,
                "de": final_de,
            })
            if len(context_window) > CONTEXT_WINDOW_SIZE:
                context_window.pop(0)

            # Log fixed Arabic
            if log_file_ar:
                append_to_file(
                    log_file_ar,
                    f"[{get_current_ts_string()}] [Batch {batch_id}] (Lat: {lat_total:.2f}s)\n{corrected_ar}\n"
                )

            # Forward to cloud worker
            if cloud_q is not None:
                cloud_q.put({
                    "range": job.get("range", (batch_id, batch_id)),
                    "id": batch_id,
                    "ar": raw_ar,
                    "final": bool(job.get("final", False)),
                })

            output_q.put({
                "type": "refined_batch",
                "id": batch_id,
                "ar_fixed": corrected_ar,
                "en_final": final_en,
                "de_final": final_de,
                "lat_s": lat_total,
            })

            if log_file_en:
                append_to_file(
                    log_file_en,
                    f"[{get_current_ts_string()}] [Batch {batch_id}] (Model: {DEFAULT_MT_MODEL} | Total Lat: {lat_total:.2f}s)\n{final_en}\n"
                )
            if log_file_de:
                append_to_file(
                    log_file_de,
                    f"[{get_current_ts_string()}] [Batch {batch_id}] (Model: {DEFAULT_MT_MODEL} | Total Lat: {lat_total:.2f}s)\n{final_de}\n"
                )

            log(f"Batch {batch_id} Done (Total Lat: {lat_total:.2f}s).")

        except queue.Empty:
            continue
        except Exception as e:
            log(f"Refinement Worker Crash: {e}")
            time.sleep(1)


# --- WORKER 2: NVIDIA NeMo FastConformer (RNNT/Transducer) ---
def transcription_stream_thread(source, config, stop_event, refine_input_q, event_q):
    """
    NVIDIA FastConformer (RNNT/Transducer Mode).
    - Mode: RNNT (smarter than CTC).
    - Context: Prepends last 0.5s of audio to current chunk to fix edge-word cuts.
    """
    import time
    import wave
    import datetime
    import numpy as np
    import queue
    import torch
    import nemo.collections.asr as nemo_asr
    import soundfile as sf
    import tempfile
    import sounddevice as sd

    stream = None
    wf = None

    # -----------------------------
    # 0) Helpers
    # -----------------------------
    def get_ts():
        return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

    log_raw = config["logs"]["raw_ar"]

    def log_to_file(text, *, path=log_raw):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except:
            pass

    # -----------------------------
    # 1) Load NVIDIA NeMo Model (RNNT Mode)
    # -----------------------------
    try:
        print(f"[{get_ts()}] Loading NVIDIA FastConformer (RNNT)...")
        asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_name="nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0"
        ).cuda()

        # Switch to RNNT decoder for better boundary handling
        asr_model.change_decoding_strategy(decoder_type="rnnt")
        asr_model.eval()

        # Disable dithering/padding for stability
        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0

        print(f"[{get_ts()}] Model Loaded (RNNT Mode).")
    except Exception as e:
        event_q.put(("error", f"Model Load Error: {e}"))
        return

    # -----------------------------
    # 2) Audio Source Setup (Strict 16kHz)
    # -----------------------------
    is_live_mic = isinstance(source, int)
    mic_queue = queue.Queue()
    TARGET_SR = 16000

    def mic_callback(indata, frames, time_info, status):
        if status:
            print(f"[MIC] {status}", flush=True)
        mic_queue.put(indata.copy())

    if is_live_mic:
        try:
            stream = sd.InputStream(
                device=source,
                channels=1,
                samplerate=TARGET_SR,
                dtype="float32",
                blocksize=int(TARGET_SR * 0.04),  # 40 ms
                callback=mic_callback,
            )
            stream.start()
        except Exception as e:
            event_q.put(("error", f"Mic Error: {e}"))
            return
    else:
        try:
            wf = wave.open(source, "rb")
            file_sr = wf.getframerate()
            if file_sr != TARGET_SR:
                print(f"Warning: File is {file_sr}Hz. Real-time resampling not implemented here.")
        except Exception as e:
            event_q.put(("error", f"File Error: {e}"))
            return

    # -----------------------------
    # 3) VAD & Buffers
    # -----------------------------
    try:
        import webrtcvad

        vad = webrtcvad.Vad(1)  # Mode 1 (less aggressive, better for long vowels)
    except Exception:
        event_q.put(("error", "webrtcvad not installed."))
        return

    PROCESS_CHUNK_SEC = 6.0
    PROCESS_SAMPLES = int(TARGET_SR * PROCESS_CHUNK_SEC)
    VAD_FRAME_MS = 30
    VAD_FRAME_SAMPLES = int(TARGET_SR * (VAD_FRAME_MS / 1000))

    OVERLAP_SEC = 0.5
    OVERLAP_SAMPLES = int(TARGET_SR * OVERLAP_SEC)
    prev_audio_tail = np.array([], dtype=np.float32)

    audio_buffer = []
    silence_counter = 0
    MAX_TRAILING_SILENCE = int(TARGET_SR * 1.0)  # hold 1s of trailing silence
    MIN_TRANSCRIPTION_LEN = int(TARGET_SR * 3.0)  # require >= 3s of audio before flush
    MAX_BUFFER_LEN = int(TARGET_SR * 10.0)  # hard cap to avoid runaway latency

    batch_buf = []
    batch_start_id = None
    batch_end_id = None
    chunk_counter = 0
    refine_every = int(config.get("refine_every", 4))

    def flush_refine_batch(force_final: bool = False):
        nonlocal batch_buf, batch_start_id, batch_end_id
        if not batch_buf:
            return
        txt = "\n".join(batch_buf).strip()
        if not txt:
            return
        job = {
            "id": batch_end_id or batch_start_id or 0,
            "range": (batch_start_id or 0, batch_end_id or batch_start_id or 0),
            "source_ar": txt,
            "ts": time.time(),
            "log_file_ar": config["logs"].get("fixed_ar"),
            "log_file_en": config["logs"].get("final_en"),
            "log_file_de": config["logs"].get("final_de"),
            "final": bool(force_final),
        }
        refine_input_q.put(job)
        batch_buf, batch_start_id, batch_end_id = [], None, None

    def transcribe_segment(audio_np: np.ndarray) -> str:
        """Save audio to temp WAV and run RNNT inference."""
        if len(audio_np) < 800:
            return ""

        mx = np.max(np.abs(audio_np))
        if mx > 0:
            audio_np = audio_np / mx

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio_np, TARGET_SR)
            try:
                hyps = asr_model.transcribe([tmp.name], batch_size=1, return_hypotheses=False)
            except Exception as e:
                print(f"Transcribe Error: {e}")
                return ""

        if not hyps:
            return ""

        res = hyps[0]
        if isinstance(res, str):
            return res
        if hasattr(res, "text"):
            return res.text
        if isinstance(res, dict) and "text" in res:
            return res["text"]
        return str(res)

    print(f"[{get_ts()}] Stream Started (RNNT).")

    # -----------------------------
    # 4) Main Loop (Stabilized)
    # -----------------------------
    while not stop_event.is_set():
        # --- Read ---
        if is_live_mic:
            try:
                frames_list = []
                while True:
                    try:
                        frames_list.append(mic_queue.get_nowait())
                    except queue.Empty:
                        break
                if not frames_list:
                    time.sleep(0.01)
                    continue
                frame_np = np.concatenate(frames_list).flatten()
            except Exception:
                continue
        else:
            raw = wf.readframes(VAD_FRAME_SAMPLES)
            if not raw:
                break
            frame_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            time.sleep(0.01)  # 10ms throttle per 30ms frame in file mode

        if len(frame_np) == 0:
            continue

        # --- VAD & Buffer ---
        idx = 0
        while idx < len(frame_np):
            chunk = frame_np[idx : idx + VAD_FRAME_SAMPLES]
            idx += VAD_FRAME_SAMPLES
            if len(chunk) < VAD_FRAME_SAMPLES:
                continue

            int16_bytes = (chunk * 32767).astype(np.int16).tobytes()
            try:
                is_speech = vad.is_speech(int16_bytes, TARGET_SR)
            except Exception:
                is_speech = True

            if is_speech:
                audio_buffer.extend(chunk)
                silence_counter = 0
            else:
                silence_counter += len(chunk)
                # keep short trailing silence to bridge gaps
                if silence_counter < MAX_TRAILING_SILENCE:
                    audio_buffer.extend(chunk)

        # --- Trigger ---
        do_transcribe = False
        buffer_len = len(audio_buffer)

        # Force flush if buffer is huge
        if buffer_len >= MAX_BUFFER_LEN:
            do_transcribe = True
        # Flush on silence only if we have enough speech
        elif silence_counter > MAX_TRAILING_SILENCE and buffer_len > MIN_TRANSCRIPTION_LEN:
            do_transcribe = True

        if do_transcribe:
            t_start = time.time()

            current_audio = np.array(audio_buffer, dtype=np.float32)
            if len(prev_audio_tail) > 0:
                full_input = np.concatenate([prev_audio_tail, current_audio])
            else:
                full_input = current_audio

            if len(current_audio) > OVERLAP_SAMPLES:
                prev_audio_tail = current_audio[-OVERLAP_SAMPLES:]
            else:
                prev_audio_tail = current_audio

            audio_buffer = []

            text = transcribe_segment(full_input)
            t_dur = time.time() - t_start

            clean = text.strip() if text else ""
            if clean:
                chunk_counter += 1
                log_msg = f"[{get_ts()}] [{chunk_counter}] (Infer: {t_dur:.2f}s) {clean}"
                print(log_msg)
                log_to_file(log_msg)

                event_q.put(("update", {"id": chunk_counter, "ar": clean, "infer_s": t_dur}))

                if batch_start_id is None:
                    batch_start_id = chunk_counter
                batch_end_id = chunk_counter
                batch_buf.append(clean)
                if len(batch_buf) >= refine_every:
                    flush_refine_batch()

    flush_refine_batch(force_final=True)
    event_q.put(("status", "stream_finished"))
    if is_live_mic and stream:
        stream.stop()
        stream.close()

def extraction_thread(video_path, wav_path, event_q):
    try:
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        event_q.put(("status", "extraction_complete"))
    except Exception as e: event_q.put(("error", f"FFMPEG Error: {e}"))

# ==========================================
# 5. MAIN UI
# ==========================================

def main():
    st.set_page_config(layout="wide", page_title="Khutbah AI")
    
    if "uid" not in st.session_state:
        st.session_state.uid = str(uuid.uuid4())[:8]
        st.session_state.chunks = [] 
        st.session_state.refined_blocks_ar = [] 
        st.session_state.refined_blocks_en = [] 
        st.session_state.refined_blocks_de = [] 
        st.session_state.cloud_blocks = []
        st.session_state.streaming = False
        st.session_state.extraction_done = False
        st.session_state.stop_event = threading.Event()
        st.session_state.last_chunk_id = 0
        
        st.session_state.refine_in = queue.Queue()
        st.session_state.refine_out = queue.Queue()
        st.session_state.event_q = queue.Queue()
        st.session_state.cloud_in = queue.Queue()
        st.session_state.cloud_out = queue.Queue()
        st.session_state.telegram_q = queue.Queue()

    if "refinement_thread_started" not in st.session_state:
        t_ref = threading.Thread(target=refinement_worker, args=(st.session_state.refine_in, st.session_state.refine_out, st.session_state.cloud_in), daemon=True)
        add_script_run_ctx(t_ref)
        t_ref.start()
        st.session_state.refinement_thread_started = True

    if "cloud_thread_started" not in st.session_state:
        t_cloud = threading.Thread(
            target=cloud_polish_worker,
            args=(st.session_state.cloud_in, st.session_state.cloud_out),
            kwargs={"target_langs": ["en", "de"], "flush_every": 8, "session_uid": st.session_state.uid},
            daemon=True,
        )
        add_script_run_ctx(t_cloud)
        t_cloud.start()
        st.session_state.cloud_thread_started = True

    if "telegram_thread_started" not in st.session_state:
        t_tg = threading.Thread(
            target=telegram_sink_worker,
            args=(st.session_state.telegram_q,),
            kwargs={"bot_token": TELEGRAM_BOT_TOKEN, "chat_id": TELEGRAM_CHAT_ID},
            daemon=True,
        )
        add_script_run_ctx(t_tg)
        t_tg.start()
        st.session_state.telegram_thread_started = True

    with st.sidebar:
        st.header("⚙️ Settings")
        
        input_mode = st.radio("Source", ["Microphone", "File Upload"], index=0)
        mic_index = None
        if input_mode == "Microphone":
            input_devices = get_input_devices()
            if input_devices:
                opts = [d[1] for d in input_devices]
                sel = st.selectbox("Device", opts)
                for idx, name in input_devices:
                    if name == sel: mic_index = idx; break
        
        st.subheader("🚀 Hardware Optimization")
        model_size = st.selectbox("Whisper Size", ["distil-large-v3", "large-v3", "medium"], index=1)
        compute_type = st.selectbox("Compute Type", ["float16", "int8_float16", "int8"], index=2)
        device = st.radio("Compute Device", ["cuda", "cpu"], index=0)
        
        refine_every = st.slider("Refine Batch Size", 2, 10, 3)
        if st.button("🔴 RESET APP"): st.session_state.stop_event.set(); st.rerun()

    st.title("🕌 Khutbah AI: Real-time Transcription")
    
    base_logs_path = os.path.join(os.getcwd(), "logs")
    st.caption(f"📂 **LOG FILES SAVING TO:** `{base_logs_path}`")

    source_to_pass = None
    if input_mode == "File Upload":
        u = st.file_uploader("Upload Audio/Video", type=["mp4","wav"])
        if u and not st.session_state.extraction_done:
            with tempfile.NamedTemporaryFile(delete=False) as t: t.write(u.read()); vp=t.name
            wp = os.path.join(tempfile.gettempdir(), f"audio_{st.session_state.uid}.wav")
            st.session_state.wav_path = wp
            with st.status("Extracting Audio..."):
                extraction_thread(vp, wp, st.session_state.event_q)
                while True:
                    t, m = st.session_state.event_q.get()
                    if t == "status": break
                    if t == "error": st.error(m); st.stop()
                st.session_state.extraction_done = True
            st.rerun()
        if st.session_state.extraction_done: source_to_pass = st.session_state.wav_path
    elif mic_index is not None:
        source_to_pass = mic_index

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

    if source_to_pass is not None:
        if not st.session_state.streaming:
            if st.button("▶️ START STREAM", type="primary", use_container_width=True):
                st.session_state.streaming = True
                st.session_state.stop_event.clear()
                
                mt_model_safe = sanitize_filename(DEFAULT_MT_MODEL)
                config = {
                    "model_size": model_size, 
                    "device": device, 
                    "compute_type": compute_type, 
                    "refine_every": refine_every, 
                    "max_window_sec": 25.0,
                    "beam_size": 2,
                    "logs": {
                        "pre_lcp_ar": get_log_path(f"log_{st.session_state.uid}_0_pre_lcp_ar.txt"),
                        "raw_ar": get_log_path(f"log_{st.session_state.uid}_1_raw_ar.txt"),
                        "fixed_ar": get_log_path(f"log_{st.session_state.uid}_2_fixed_ar.txt"),
                        "final_en": get_log_path(f"log_{st.session_state.uid}_{mt_model_safe}_en.txt"),
                        "final_de": get_log_path(f"log_{st.session_state.uid}_{mt_model_safe}_de.txt")
                    }
                }

                t2 = threading.Thread(target=transcription_stream_thread, args=(source_to_pass, config, st.session_state.stop_event, st.session_state.refine_in, st.session_state.event_q), daemon=True)
                add_script_run_ctx(t2)
                t2.start()
                st.rerun()
        else:
            if st.button("⏹️ STOP STREAM", type="secondary", use_container_width=True):
                st.session_state.stop_event.set()
                st.session_state.streaming = False
                st.rerun()

    if not st.session_state.streaming:
        box_raw.text_area("Raw", value="\n\n".join([c['ar'] for c in st.session_state.chunks]), height=600, key="static_raw")
        box_fixed.text_area("Refined Arabic", value="\n\n".join(st.session_state.refined_blocks_ar), height=600, key="static_fixed")
        box_final_en.text_area("Refined English", value="\n\n".join(st.session_state.refined_blocks_en), height=600, key="static_final_en")
        box_final_de.text_area("Refined German", value="\n\n".join(st.session_state.refined_blocks_de), height=600, key="static_final_de")
        box_cloud.text_area("Cloud Translation", value="\n\n".join(st.session_state.cloud_blocks), height=600, key="static_cloud")

    if st.session_state.streaming:
        while not st.session_state.stop_event.is_set():
            has_data = False
            try:
                while True:
                    t, p = st.session_state.event_q.get_nowait()
                    if t == "update":
                        st.session_state.chunks.append(p)
                        st.session_state.last_chunk_id = p.get("id", st.session_state.last_chunk_id)
                        has_data = True
                        # Fan out raw Arabic chunk to Telegram
                        infer_info = f" (Infer: {p['infer_s']:.2f}s)" if "infer_s" in p else ""
                        st.session_state.telegram_q.put({
                            "text": f"<b>[ASR #{p.get('id', '?')}]{infer_info}</b>\n{p['ar']}"
                        })
                    elif t == "error":
                        st.error(f"Error: {p}")
                        st.session_state.stop_event.set()
                        break
                    elif t == "status" and p == "stream_finished":
                        st.session_state.stop_event.set()
                        break
            except queue.Empty:
                pass

            try:
                while True:
                    p = st.session_state.refine_out.get_nowait()
                    if p["type"] == "refined_batch":
                        st.session_state.refined_blocks_ar.append(f"[{p['id']}] {p['ar_fixed']}")
                        st.session_state.refined_blocks_en.append(f"[{p['id']}] {p['en_final']}")
                        st.session_state.refined_blocks_de.append(f"[{p['id']}] {p['de_final']}")
                        has_data = True
                        # Fan out to Telegram
                        _lat_info = f" (Lat: {p['lat_s']:.2f}s)" if "lat_s" in p else ""
                        st.session_state.telegram_q.put({
                            "text": (
                                f"<b>[Ollama #{p['id']}]{_lat_info}</b>\n"
                                f"<b>AR:</b> {p['ar_fixed']}\n"
                                f"<b>EN:</b> {p['en_final']}\n"
                                f"<b>DE:</b> {p['de_final']}"
                            )
                        })
            except queue.Empty: pass

            try:
                while True:
                    p = st.session_state.cloud_out.get_nowait()
                    st.session_state.cloud_blocks.append(
                        f"[{p['range'][0]}–{p['range'][1]}] ({p['lang']}, {p['model']})\n{p['text']}"
                    )
                    has_data = True
                    # Fan out to Telegram
                    _clat = f" (Lat: {p['lat_s']:.2f}s)" if "lat_s" in p else ""
                    st.session_state.telegram_q.put({
                        "text": (
                            f"<b>[Cloud {p['range'][0]}–{p['range'][1]}] ({p['lang']}, {p['model']}){_clat}</b>\n"
                            f"{p['text']}"
                        )
                    })
            except queue.Empty:
                pass
            
            if has_data:
                iter_id = str(uuid.uuid4())[:8] 
                box_raw.text_area("Raw", value="\n\n".join([c['ar'] for c in st.session_state.chunks]), height=600, key=f"raw_{iter_id}")
                box_fixed.text_area("Refined Arabic", value="\n\n".join(st.session_state.refined_blocks_ar), height=600, key=f"fixed_{iter_id}")
                box_final_en.text_area("Refined English", value="\n\n".join(st.session_state.refined_blocks_en), height=600, key=f"final_en_{iter_id}")
                box_final_de.text_area("Refined German", value="\n\n".join(st.session_state.refined_blocks_de), height=600, key=f"final_de_{iter_id}")
                box_cloud.text_area("Cloud Translation", value="\n\n".join(st.session_state.cloud_blocks), height=600, key=f"cloud_{iter_id}")

            time.sleep(0.2) 
        if "cloud_in" in st.session_state and st.session_state.cloud_in:
            try:
                st.session_state.cloud_in.put({"id": st.session_state.last_chunk_id, "ar": "", "final": True})
            except Exception:
                pass
        st.rerun()

if __name__ == "__main__":
    main()
