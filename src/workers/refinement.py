"""
Refinement worker for Arabic text correction and translation.

Uses local LLM (OLLAMA) to fix ASR errors and translate to English/German.
"""

import time
import queue
from typing import List, Dict, Optional

from ..config import DEFAULT_MT_MODEL, CONTEXT_WINDOW_SIZE
from ..utils import log, get_current_ts_string, append_to_file
from ..arabic_utils import arabic_ratio, glossary_block
from ..api.ollama import robust_ollama_call, parse_tagged_response


def refinement_worker(
    input_q: queue.Queue,
    output_q: queue.Queue,
    cloud_q: Optional[queue.Queue] = None,
    config: Optional[dict] = None,
) -> None:
    """
    Worker thread for refining Arabic text and translating.

    Consumes jobs from input_q, processes them, and puts results in output_q.
    Optionally forwards to cloud_q for high-quality cloud processing.

    Args:
        input_q: Queue of jobs with source_ar text
        output_q: Queue for refined results
        cloud_q: Optional queue to forward to cloud worker
        config: Optional configuration dict
    """
    log("Refinement Worker: STARTING")

    # Sliding window of previous results for context
    context_window: List[Dict[str, str]] = []

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
            ctx_lines = [
                entry.get("fixed_ar", entry.get("raw", ""))
                for entry in context_window
            ]
            parts.append(
                "### PREVIOUS CONTEXT (read-only, do NOT include in output):\n"
                + "\n".join(ctx_lines)
            )
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

            # If the speaker switched language (not Arabic), skip processing
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

            # Single unified fix + translate call
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

            # Fallback if parse failed
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
