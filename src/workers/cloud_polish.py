"""
Cloud polishing worker using Groq for high-quality translation.

Batches Arabic text and sends to Groq for professional-grade
correction and multi-language translation.
"""

import time
import queue
from typing import List, Dict, Optional

from ..config import CONTEXT_WINDOW_SIZE
from ..utils import log, get_current_ts_string, append_to_file, get_log_path, sanitize_filename
from ..api.ollama import parse_tagged_response
from ..api.groq_client import groq_chat_with_fallback, get_groq_client, GROQ_MODEL_FALLBACK


def cloud_polish_worker(
    cloud_in_q: queue.Queue,
    cloud_out_q: queue.Queue,
    *,
    target_langs: Optional[List[str]] = None,
    flush_every: int = 8,
    session_uid: str = "",
) -> None:
    """
    Worker thread for cloud-based polishing and translation.

    Batches input text and sends to Groq for high-quality processing.

    Args:
        cloud_in_q: Queue of Arabic text batches
        cloud_out_q: Queue for translated results
        target_langs: Target languages (default: ["en", "de"])
        flush_every: Number of items to batch before sending
        session_uid: Session identifier for logging
    """
    groq_client = get_groq_client()

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
            parts.append(
                "### PREVIOUS CONTEXT (read-only, do NOT include in output):\n"
                + "\n".join(ctx_lines)
            )
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

            should_flush = (
                (len(buf) >= flush_every)
                or bool(item.get("final", False))
                or time_due
            )
        else:
            should_flush = time_due

        if not should_flush or not buf:
            continue

        end_id = int(end_id) if end_id is not None else (
            start_id + len(buf) - 1 if start_id is not None else 0
        )
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
                    log_name_ar = (
                        f"log_{session_uid}_cloud_{safe_model}_ar.txt"
                        if session_uid
                        else f"log_cloud_{safe_model}_ar.txt"
                    )
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
                    log_name_en = (
                        f"log_{session_uid}_cloud_{safe_model}_en.txt"
                        if session_uid
                        else f"log_cloud_{safe_model}_en.txt"
                    )
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
                    log_name_de = (
                        f"log_{session_uid}_cloud_{safe_model}_de.txt"
                        if session_uid
                        else f"log_cloud_{safe_model}_de.txt"
                    )
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
