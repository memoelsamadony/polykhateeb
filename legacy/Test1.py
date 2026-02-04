# app.py
import os
import time
import uuid
import queue
import wave
import tempfile
import subprocess
import threading
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import streamlit as st
import numpy as np  # noqa: F401
from faster_whisper import WhisperModel

# ------------------------
# Logging helpers
# ------------------------
def log(msg: str) -> None:
    """Lightweight stdout logger with flush to ensure visibility in Streamlit logs."""
    print(msg, flush=True)

def append_line(path: str, text: str) -> None:
    """Append a line to a log file (best-effort)."""
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception as e:
        log(f"[LOGFILE] failed to append to {path}: {e}")

def ensure_log_file(path: str) -> None:
    """Make sure a log file exists; create parent dirs and touch file."""
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch(exist_ok=True)
    except Exception as e:
        log(f"[LOGFILE] failed to init {path}: {e}")

# ------------------------
# Optional: streamlit autorefresh
# ------------------------
try:
    from streamlit_autorefresh import st_autorefresh
    _AUTOREFRESH_OK = True
except Exception:
    _AUTOREFRESH_OK = False

# ------------------------
# Optional: Gemini
# ------------------------
try:
    import google.generativeai as genai
    _GEMINI_OK = True
except Exception:
    _GEMINI_OK = False

# ------------------------
# Optional: Local translation (MarianMT)
# ------------------------
try:
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    _TRANSFORMERS_OK = True
except Exception:
    torch = None  # IMPORTANT: avoid NameError later
    MarianMTModel = None
    MarianTokenizer = None
    _TRANSFORMERS_OK = False

# ---------------- CONFIG ----------------
DEFAULT_GEMINI_MODEL = "models/gemini-2.5-flash"
WHISPER_MODEL_SIZE = "large-v2"
WHISPER_CACHE_DIR = str(Path.home() / ".cache" / "faster-whisper")

# Whisper decoding defaults (accuracy-focused)
WHISPER_BEAM_DEFAULT = 7
WHISPER_BEAM_MAX = 8
WHISPER_BEST_OF_DEFAULT = 7
WHISPER_BEST_OF_MAX = 8

# Streaming chunking defaults (accept higher latency for more context)
CHUNK_SEC_DEFAULT = 30.0
CHUNK_SEC_MAX = 90.0
OVERLAP_SEC_DEFAULT = 1.0

MARIAN_AR_EN = "Helsinki-NLP/opus-mt-ar-en"
MARIAN_AR_DE = "Helsinki-NLP/opus-mt-ar-de"

DEFAULT_GEMINI_REFINE_EVERY_CHUNKS = 15
GEMINI_CONTEXT_CHARS = 2200
GEMINI_TRANSLATION_CONTEXT_CHARS = 2200

CUDA_HOUSEKEEPING_EVERY = 10

INITIAL_PROMPT_ISLAMIC = (
    "This is an Arabic Islamic lecture. Preserve Islamic terms faithfully: "
    "Allah, Qur'an, Sunnah, hadith, salah, zakah, sawm, Hajj, Umrah, tawhid."
)

# ---------------- Islamic glossary / term preservation ----------------
GLOSSARY: Dict[str, Dict[str, str]] = {
    "الله": {"English": "Allah", "German": "Allah"},
    "القرآن": {"English": "the Qur'an", "German": "der Qur'an"},
    "قرآن": {"English": "Qur'an", "German": "Qur'an"},
    "حديث": {"English": "hadith", "German": "Hadith"},
    "السنة": {"English": "the Sunnah", "German": "die Sunnah"},
    "سنة": {"English": "Sunnah", "German": "Sunnah"},
    "صلاة": {"English": "salah (prayer)", "German": "Salāh (Gebet)"},
    "الصلاة": {"English": "the salah (prayer)", "German": "die Salāh (das Gebet)"},
    "زكاة": {"English": "zakah (alms)", "German": "Zakāh (Almosenabgabe)"},
    "الزكاة": {"English": "the zakah (alms)", "German": "die Zakāh (Almosenabgabe)"},
    "صوم": {"English": "sawm (fasting)", "German": "Sawm (Fasten)"},
    "الصوم": {"English": "the sawm (fasting)", "German": "das Sawm (Fasten)"},
    "حج": {"English": "Hajj (pilgrimage)", "German": "Hadsch (Pilgerfahrt)"},
    "الحج": {"English": "the Hajj (pilgrimage)", "German": "der Hadsch (Pilgerfahrt)"},
    "عمرة": {"English": "Umrah", "German": "ʿUmrah"},
    "إحرام": {"English": "ihram", "German": "Ihrām"},
    "فتوى": {"English": "fatwa", "German": "Fatwā"},
    "حرام": {"English": "haram (forbidden)", "German": "haram (verboten)"},
    "حلال": {"English": "halal (permissible)", "German": "halal (erlaubt)"},
    "توحيد": {"English": "tawhid (monotheism)", "German": "Tawhīd (Einheitsglaube)"},
}

def protect_terms(text: str, target_lang: str) -> Tuple[str, Dict[str, str]]:
    placeholder_map: Dict[str, str] = {}
    out = text
    keys = sorted(GLOSSARY.keys(), key=len, reverse=True)
    idx = 0
    for k in keys:
        if k in out:
            idx += 1
            ph = f"<<TERM_{idx}>>"
            placeholder_map[ph] = GLOSSARY[k].get(target_lang, k)
            out = out.replace(k, ph)
    return out, placeholder_map

def restore_terms(text: str, placeholder_map: Dict[str, str]) -> str:
    out = text
    for ph, val in placeholder_map.items():
        out = out.replace(ph, val)
    return out

# ---------------- Per-session queue (stable across reruns) ----------------
def get_updates_q() -> "queue.Queue[Tuple[str, dict]]":
    if "updates_q" not in st.session_state:
        st.session_state.updates_q = queue.Queue()
    return st.session_state.updates_q

# ---------------- Whisper / MT loaders (MAIN THREAD ONLY) ----------------
@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    # local-only first, then allow download
    try:
        return WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=WHISPER_CACHE_DIR,
            local_files_only=True,
        )
    except Exception:
        return WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=WHISPER_CACHE_DIR,
            local_files_only=False,
        )

@st.cache_resource(show_spinner=False)
def load_local_mt(target_lang: str):
    if not _TRANSFORMERS_OK:
        return None
    model_name = MARIAN_AR_EN if target_lang == "English" else MARIAN_AR_DE
    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

    log(f"[MT] Loading MarianMT for {target_lang} on {device}...")
    tok = MarianTokenizer.from_pretrained(model_name)
    mdl = MarianMTModel.from_pretrained(model_name)

    if device == "cuda":
        try:
            mdl = mdl.to("cuda")
            log("[MT] MarianMT loaded on CUDA")
        except Exception as e:
            log(f"[MT] Failed to move MarianMT to CUDA: {e}; falling back to CPU")
            device = "cpu"
            mdl = mdl.to("cpu")
    else:
        log("[MT] MarianMT using CPU (no CUDA available)")

    mdl.eval()
    return tok, mdl

def local_translate_delta_with_model(tok, mdl, delta_ar: str, target_lang: str) -> str:
    if not _TRANSFORMERS_OK or torch is None:
        return ""
    if not delta_ar.strip():
        return ""

    protected, ph_map = protect_terms(delta_ar, target_lang)
    on_cuda = torch.cuda.is_available() and next(mdl.parameters()).is_cuda

    batch = tok([protected], return_tensors="pt", truncation=True, max_length=512)
    if on_cuda:
        batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}

    with torch.inference_mode():
        if on_cuda:
            with torch.autocast("cuda", dtype=torch.float16):
                out = mdl.generate(**batch, max_length=256, num_beams=1)
        else:
            out = mdl.generate(**batch, max_length=256, num_beams=1)

    text = tok.batch_decode(out, skip_special_tokens=True)[0]

    # cleanup locals (important)
    del batch, out
    return restore_terms(text.strip(), ph_map)

def cuda_housekeeping(every_n_chunks: int, chunk_idx: int):
    if not _TRANSFORMERS_OK or torch is None or (not torch.cuda.is_available()):
        return
    if every_n_chunks <= 0:
        return
    if chunk_idx % every_n_chunks == 0:
        torch.cuda.empty_cache()
        log(
            f"[CUDA] Housekeeping at chunk {chunk_idx}: "
            f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB "
            f"allocated={torch.cuda.memory_allocated()/1e9:.2f}GB"
        )

# ---------------- Gemini (MAIN THREAD ONLY) ----------------
def configure_gemini() -> bool:
    if not _GEMINI_OK:
        return False
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return False
    genai.configure(api_key=key)
    return True

def gemini_refine_translation(
    transcript_tail: str,
    draft_translation_tail: str,
    target_lang: str,
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> str:
    if not configure_gemini():
        return "(Gemini not configured)"

    rules = f"""
You are a careful translator. Translate Arabic -> {target_lang}.

IMPORTANT RELIGIOUS CONTEXT RULES:
- Preserve meaning, do not add or remove religious content.
- Keep Islamic terms consistent with the glossary.
- If a term is in the glossary, use the glossary rendering.
""".strip()

    glossary_lines = [
        f"{ar} => {mapping.get(target_lang, mapping.get('English',''))}"
        for ar, mapping in GLOSSARY.items()
    ]
    glossary_block = "\n".join(glossary_lines[:80])

    prompt = f"""{rules}

GLOSSARY (use consistently when relevant):
{glossary_block}

ARABIC TRANSCRIPT (recent context):
{transcript_tail}

DRAFT TRANSLATION (recent context):
{draft_translation_tail}

Return ONLY the improved translation text (no explanations).
"""

    model = genai.GenerativeModel(model_name=model_name)
    log(
        f"[GEMINI] Request prompt_len={len(prompt)} "
        f"transcript_tail_len={len(transcript_tail)} "
        f"translation_tail_len={len(draft_translation_tail)}"
    )
    resp = model.generate_content(prompt)
    text = (resp.text or "").strip()
    log(f"[GEMINI] Response len={len(text)} preview='{text[:240].replace(chr(10),' ')}...'")
    return text

# ---------------- Audio extraction ----------------
def video_to_wav(video_path: str, wav_path: str, sr: int = 16000) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-c:a",
        "pcm_s16le",
        wav_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def extraction_worker(updates_q: "queue.Queue[Tuple[str, dict]]", video_path: str, wav_path: str):
    try:
        log(f"[EXTRACT] Start ffmpeg: video={video_path}, wav={wav_path}, qid={id(updates_q)}")
        video_to_wav(video_path, wav_path, sr=16000)
        log(f"[EXTRACT] Success: wav={wav_path}")
        updates_q.put(("extraction_done", {}))
    except Exception as e:
        log(f"[EXTRACT] Failed: {e}")
        updates_q.put(("extraction_error", {"error": str(e)}))

# ---------------- Streaming helpers ----------------
def _normalize_words(s: str) -> List[str]:
    return [w for w in s.strip().split() if w]

def _dedupe_append(full_text: str, new_text: str, max_overlap_words: int = 50) -> str:
    if not new_text:
        return full_text.strip()

    full_words = _normalize_words(full_text)
    new_words = _normalize_words(new_text)

    if not full_words:
        return " ".join(new_words).strip()

    max_k = min(max_overlap_words, len(full_words), len(new_words))
    overlap_k = 0
    for k in range(max_k, 0, -1):
        if full_words[-k:] == new_words[:k]:
            overlap_k = k
            break

    appended = new_words[overlap_k:] if overlap_k else new_words
    return " ".join(full_words + appended).strip()

def transcribe_file_once_with_model(
    model: WhisperModel,
    path: str,
    beam_size: int,
    best_of: int,
    initial_prompt: str,
) -> str:
    segments, _ = model.transcribe(
        path,
        beam_size=beam_size,
        best_of=min(max(best_of, 1), WHISPER_BEST_OF_MAX),
        vad_filter=True,
        no_speech_threshold=0.6,
        language="ar",
        initial_prompt=initial_prompt or None,
        condition_on_previous_text=True,
    )
    return " ".join([s.text for s in segments]).strip()

def stream_worker(
    updates_q: "queue.Queue[Tuple[str, dict]]",
    stop_event: threading.Event,
    whisper_model: WhisperModel,
    tok,
    mt_model,
    target_lang: str,
    wav_path: str,
    beam_size: int,
    best_of: int,
    chunk_sec: float,
    overlap_sec: float,
    realtime_sleep: bool,
):
    log(f"[STREAM] start qid={id(updates_q)}")
    full_text = ""
    rolling_prompt = INITIAL_PROMPT_ISLAMIC
    chunk_count = 0

    try:
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            if sw != 2:
                updates_q.put(("stream_done", {}))
                return

            hop_frames = max(1, int(sr * chunk_sec))
            overlap_frames = max(0, int(sr * overlap_sec))
            prev_overlap_bytes = b""

            while not stop_event.is_set():
                hop_bytes = wf.readframes(hop_frames)
                if not hop_bytes:
                    break

                window_bytes = (prev_overlap_bytes + hop_bytes) if overlap_frames > 0 else hop_bytes

                if overlap_frames > 0:
                    bytes_per_frame = ch * sw
                    need = overlap_frames * bytes_per_frame
                    prev_overlap_bytes = window_bytes[-need:] if len(window_bytes) >= need else window_bytes
                else:
                    prev_overlap_bytes = b""

                tmp_chunk = os.path.join(tempfile.gettempdir(), f"chunk_{time.time_ns()}.wav")
                with wave.open(tmp_chunk, "wb") as out:
                    out.setnchannels(ch)
                    out.setsampwidth(2)
                    out.setframerate(sr)
                    out.writeframes(window_bytes)

                try:
                    prompt_tail = (INITIAL_PROMPT_ISLAMIC + " " + rolling_prompt).strip()[:900]
                    chunk_text = transcribe_file_once_with_model(
                        whisper_model,
                        tmp_chunk,
                        beam_size=beam_size,
                        best_of=best_of,
                        initial_prompt=prompt_tail,
                    )
                finally:
                    try:
                        os.remove(tmp_chunk)
                    except Exception:
                        pass

                if chunk_text:
                    before = full_text
                    full_text = _dedupe_append(full_text, chunk_text, max_overlap_words=50)
                    delta = full_text[len(before):].strip() if full_text.startswith(before) else chunk_text

                    rolling_prompt = full_text[-800:]

                    local_delta = ""
                    if tok is not None and mt_model is not None:
                        local_delta = local_translate_delta_with_model(tok, mt_model, delta, target_lang)

                    cuda_housekeeping(every_n_chunks=CUDA_HOUSEKEEPING_EVERY, chunk_idx=chunk_count + 1)

                    updates_q.put(
                        (
                            "chunk",
                            {
                                "full_transcript": full_text,
                                "delta_ar": delta,
                                "local_delta": local_delta,
                                "chunk_idx": chunk_count,
                            },
                        )
                    )

                    chunk_count += 1
                    log(
                        f"[STREAM] Chunk {chunk_count}: delta_ar_len={len(delta)}, local_delta_len={len(local_delta)}\n"
                        f" delta_ar='{delta[:160]}'\n local_delta='{local_delta[:160]}'"
                    )

                if realtime_sleep:
                    time.sleep(chunk_sec)

    except Exception as e:
        updates_q.put(("stream_error", {"error": str(e)}))
        log(f"[STREAM] Error: {e}")
    finally:
        log("[STREAM] Worker finishing, signaling stream_done")
        updates_q.put(("stream_done", {}))

# ---------------- Session state ----------------
def ensure_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "full_transcript" not in st.session_state:
        st.session_state.full_transcript = ""
    if "full_translation_local" not in st.session_state:
        st.session_state.full_translation_local = ""
    if "full_translation_refined" not in st.session_state:
        st.session_state.full_translation_refined = ""

    if "last_delta_ar" not in st.session_state:
        st.session_state.last_delta_ar = ""
    if "last_delta_local" not in st.session_state:
        st.session_state.last_delta_local = ""

    if "chunk_since_last_refine" not in st.session_state:
        st.session_state.chunk_since_last_refine = 0
    if "maybe_refine_due" not in st.session_state:
        st.session_state.maybe_refine_due = False

    if "streaming_running" not in st.session_state:
        st.session_state.streaming_running = False
    if "stop_event" not in st.session_state:
        st.session_state.stop_event = threading.Event()

    if "extraction_in_progress" not in st.session_state:
        st.session_state.extraction_in_progress = False
    if "extraction_done" not in st.session_state:
        st.session_state.extraction_done = False
    if "extraction_error" not in st.session_state:
        st.session_state.extraction_error = ""

    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "wav_path" not in st.session_state:
        st.session_state.wav_path = None
    if "upload_key" not in st.session_state:
        st.session_state.upload_key = None

    # Gemini control (main thread only)
    if "gemini_enabled" not in st.session_state:
        st.session_state.gemini_enabled = False
    if "gemini_last_call_ts" not in st.session_state:
        st.session_state.gemini_last_call_ts = 0.0
    if "gemini_block_until_ts" not in st.session_state:
        st.session_state.gemini_block_until_ts = 0.0
    if "gemini_last_error" not in st.session_state:
        st.session_state.gemini_last_error = ""
    if "gemini_refine_every_chunks" not in st.session_state:
        st.session_state.gemini_refine_every_chunks = DEFAULT_GEMINI_REFINE_EVERY_CHUNKS

    if "start_stream_request" not in st.session_state:
        st.session_state.start_stream_request = False
    if "chunk_total" not in st.session_state:
        st.session_state.chunk_total = 0

    # log file paths (per-session temp files)
    if "transcript_log_path" not in st.session_state:
        st.session_state.transcript_log_path = os.path.join(
            tempfile.gettempdir(), f"transcript_{st.session_state.session_id}.txt"
        )
    if "translation_log_path" not in st.session_state:
        st.session_state.translation_log_path = os.path.join(
            tempfile.gettempdir(), f"translation_{st.session_state.session_id}.txt"
        )
    ensure_log_file(st.session_state.transcript_log_path)
    ensure_log_file(st.session_state.translation_log_path)

def drain_updates(updates_q: "queue.Queue[Tuple[str, dict]]") -> bool:
    changed = False
    while True:
        try:
            kind, payload = updates_q.get_nowait()
        except queue.Empty:
            break

        changed = True
        log(f"[DRAIN] kind={kind} payload_keys={list(payload.keys())}")

        if kind == "extraction_done":
            st.session_state.extraction_done = True
            st.session_state.extraction_in_progress = False
            st.session_state.extraction_error = ""
            log("[DRAIN] extraction_done -> True")

        elif kind == "extraction_error":
            st.session_state.extraction_done = False
            st.session_state.extraction_in_progress = False
            st.session_state.extraction_error = payload.get("error", "Unknown error")
            log(f"[DRAIN] extraction_error -> {st.session_state.extraction_error}")

        elif kind == "chunk":
            st.session_state.full_transcript = payload["full_transcript"]
            st.session_state.last_delta_ar = payload.get("delta_ar", "")

            local_delta = (payload.get("local_delta") or "").strip()
            if local_delta:
                if st.session_state.full_translation_local and not st.session_state.full_translation_local.endswith((" ", "\n")):
                    st.session_state.full_translation_local += " "
                st.session_state.full_translation_local += local_delta
            st.session_state.last_delta_local = local_delta

            st.session_state.chunk_total += 1

            append_line(st.session_state.transcript_log_path, st.session_state.last_delta_ar)
            if local_delta:
                append_line(st.session_state.translation_log_path, st.session_state.last_delta_local)

            # every 90 chunks dump full concatenated text
            if st.session_state.chunk_total % 90 == 0:
                append_line(
                    st.session_state.transcript_log_path,
                    "\n---- FULL TRANSCRIPT @90 ----\n" + st.session_state.full_transcript + "\n---- END ----\n",
                )
                append_line(
                    st.session_state.translation_log_path,
                    "\n---- FULL TRANSLATION @90 ----\n"
                    + st.session_state.full_translation_local
                    + "\n---- END ----\n",
                )

            st.session_state.chunk_since_last_refine += 1
            if st.session_state.chunk_since_last_refine >= st.session_state.gemini_refine_every_chunks:
                st.session_state.maybe_refine_due = True
                log(
                    "[DRAIN] Gemini refine due by chunks: "
                    f"chunk_since_last_refine={st.session_state.chunk_since_last_refine} "
                    f"threshold={st.session_state.gemini_refine_every_chunks}"
                )

            log(
                "[DRAIN] chunk applied: transcript_len=%s local_len=%s last_delta_len=%s"
                % (
                    len(st.session_state.full_transcript),
                    len(st.session_state.full_translation_local),
                    len(st.session_state.last_delta_ar),
                )
            )

            # Safe CUDA debug logging (only if torch exists)
            if _TRANSFORMERS_OK and torch is not None and torch.cuda.is_available():
                log(
                    f"[CUDA] after chunk number: {st.session_state.chunk_total} "
                    f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB "
                    f"allocated={torch.cuda.memory_allocated()/1e9:.2f}GB"
                )

        elif kind == "stream_done":
            st.session_state.streaming_running = False
            log("[DRAIN] stream_done -> streaming_running=False")

        elif kind == "stream_error":
            st.session_state.streaming_running = False
            st.session_state.extraction_error = payload.get("error", "stream error")
            log(f"[DRAIN] stream_error -> {st.session_state.extraction_error}")

    return changed

# ---------------- Streamlit UI ----------------
def app_ui():
    st.set_page_config(page_title="Realtime Transcribe & Translate", layout="wide")
    ensure_state()
    updates_q = get_updates_q()

    # Apply queued updates first
    _ = drain_updates(updates_q)

    # Refresh frequently to pick up background queue updates
    if _AUTOREFRESH_OK:
        st_autorefresh(interval=800, key=f"ui_refresh_{st.session_state.session_id}")
    else:
        st.caption("Tip: pip install streamlit-autorefresh for smoother real-time updates.")

    st.title("Realtime Transcribe & Translate (Fixed extraction + realtime chunk UI)")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        target_lang = st.selectbox("Translate to", ["English", "German"], index=0)
        device = st.radio("Whisper device", ["cpu", "cuda"], index=1, horizontal=True)
        model_size = st.selectbox(
            "Whisper model",
            ["tiny", "base", "small", "medium", "large-v2"],
            index=["tiny", "base", "small", "medium", "large-v2"].index(WHISPER_MODEL_SIZE),
        )
        compute_type = st.selectbox("Whisper compute", ["int8_float16", "float16", "int8", "default"], index=0)
        beam_size = st.slider("Beam size", 1, WHISPER_BEAM_MAX, WHISPER_BEAM_DEFAULT, 1)
        best_of = st.slider("Best of", 1, WHISPER_BEST_OF_MAX, WHISPER_BEST_OF_DEFAULT, 1)
        chunk_sec = st.slider("Chunk hop (s)", 5.0, CHUNK_SEC_MAX, CHUNK_SEC_DEFAULT, 0.5)
        overlap_sec = st.slider("Overlap (s)", 0.0, 3.0, OVERLAP_SEC_DEFAULT, 0.1)
        realtime_sleep = st.checkbox("Simulate realtime (sleep)", value=True)

        st.divider()
        st.subheader("Gemini refinement (optional)")
        st.caption("Runs in MAIN thread, throttled. Doesn’t burn quota every chunk.")
        st.session_state.gemini_enabled = st.checkbox("Enable Gemini refinement", value=st.session_state.gemini_enabled)
        st.session_state.gemini_refine_every_chunks = st.slider(
            "Refine every N chunks", 5, 50, st.session_state.gemini_refine_every_chunks, 1
        )

        if _GEMINI_OK:
            gemini_key = st.text_input("GEMINI_API_KEY", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
            if gemini_key.strip():
                os.environ["GEMINI_API_KEY"] = gemini_key.strip()

        st.divider()
        if st.button("Warm-load Whisper"):
            with st.spinner("Loading Whisper..."):
                _ = load_whisper_model(model_size, device, compute_type)
            st.success("Whisper loaded & cached.")

        if _TRANSFORMERS_OK and st.button("Warm-load Local Translator"):
            with st.spinner("Loading local MT model..."):
                _ = load_local_mt(target_lang)
            st.success("Local translator loaded & cached.")

    # Main panels
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transcription (Arabic)")
        st.code(st.session_state.full_transcript or "(empty)", language="text")
    with col2:
        st.subheader(f"Translation (Local, {target_lang})")
        st.code(st.session_state.full_translation_local or "(empty)", language="text")

    st.subheader("Latest chunk deltas")
    delta_col1, delta_col2 = st.columns(2)
    with delta_col1:
        st.caption("Arabic delta (last chunk)")
        st.code(st.session_state.last_delta_ar or "(no chunks yet)", language="text")
    with delta_col2:
        st.caption(f"Local translation delta ({target_lang})")
        st.code(st.session_state.last_delta_local or "(no chunks yet)", language="text")

    st.subheader("Running totals")
    totals_col1, totals_col2 = st.columns(2)
    with totals_col1:
        st.metric("Transcript length (chars)", len(st.session_state.full_transcript))
    with totals_col2:
        st.metric("Local translation length (chars)", len(st.session_state.full_translation_local))

    st.caption(
        f"Queue size: {updates_q.qsize()} | chunk_since_last_refine={st.session_state.chunk_since_last_refine} | "
        f"maybe_refine_due={st.session_state.maybe_refine_due} | streaming_running={st.session_state.streaming_running} | "
        f"extraction_done={st.session_state.extraction_done} | cuda_housekeeping_every={CUDA_HOUSEKEEPING_EVERY} chunks"
    )

    st.subheader("Refined Translation (Gemini, throttled)")
    st.code(st.session_state.full_translation_refined or "(empty)", language="text")
    if st.session_state.gemini_last_error:
        st.warning(f"Gemini error: {st.session_state.gemini_last_error[:260]}...")

    st.divider()
    st.subheader("Video -> Extract -> Stream")

    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "mkv", "webm", "avi"])
    if uploaded is not None:
        upload_key = (uploaded.name, uploaded.size)

        # New upload => reset + save file
        if st.session_state.upload_key != upload_key:
            # cleanup old temp files
            for p in [st.session_state.video_path, st.session_state.wav_path]:
                if p and isinstance(p, str) and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

            st.session_state.upload_key = upload_key
            st.session_state.full_transcript = ""
            st.session_state.full_translation_local = ""
            st.session_state.full_translation_refined = ""
            st.session_state.gemini_last_error = ""
            st.session_state.last_delta_ar = ""
            st.session_state.last_delta_local = ""
            st.session_state.chunk_since_last_refine = 0
            st.session_state.maybe_refine_due = False
            st.session_state.streaming_running = False

            st.session_state.stop_event.set()
            st.session_state.stop_event = threading.Event()

            st.session_state.extraction_done = False
            st.session_state.extraction_in_progress = False
            st.session_state.extraction_error = ""

            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
                tmp.write(uploaded.getbuffer())
                st.session_state.video_path = tmp.name

            st.session_state.wav_path = os.path.join(tempfile.gettempdir(), f"extracted_{time.time_ns()}.wav")
            log(f"[APP] New upload saved: video={st.session_state.video_path}, wav={st.session_state.wav_path}")

        st.video(st.session_state.video_path)

        # Start extraction ONCE
        if (
            (not st.session_state.extraction_done)
            and (not st.session_state.extraction_in_progress)
            and (not st.session_state.extraction_error)
        ):
            st.session_state.extraction_in_progress = True
            log("[APP] Starting extraction thread")
            threading.Thread(
                target=extraction_worker,
                args=(updates_q, st.session_state.video_path, st.session_state.wav_path),
                daemon=True,
            ).start()
            st.rerun()

        # Status
        if st.session_state.extraction_in_progress:
            st.info("🎧 Extracting audio with ffmpeg...")
        if st.session_state.extraction_error:
            st.error(f"⚠️ Extraction failed: {st.session_state.extraction_error}")

        # Start streaming button
        if st.session_state.extraction_done and (not st.session_state.streaming_running):
            st.success("✅ Audio extracted. Ready to stream.")
            if st.button("🎬 Start streaming transcription", key="start_stream_btn"):
                st.session_state.start_stream_request = True
                log("[APP] Start streaming requested")

        # Honor start request (survives autorefresh reruns)
        if (
            st.session_state.start_stream_request
            and st.session_state.extraction_done
            and (not st.session_state.streaming_running)
        ):
            st.session_state.start_stream_request = False
            log("[APP] Starting streaming worker now")

            if _TRANSFORMERS_OK and torch is not None and torch.cuda.is_available():
                log(
                    f"[CUDA] pre-load reserved={torch.cuda.memory_reserved()/1e9:.2f}GB "
                    f"allocated={torch.cuda.memory_allocated()/1e9:.2f}GB"
                )

            # preload models in MAIN thread
            with st.spinner("Loading models (one-time)..."):
                log("[APP] Loading Whisper...")
                whisper_model = load_whisper_model(model_size, device, compute_type)
                log("[APP] Whisper loaded.")

                if _TRANSFORMERS_OK and torch is not None and torch.cuda.is_available():
                    log(
                        f"[CUDA] after Whisper reserved={torch.cuda.memory_reserved()/1e9:.2f}GB "
                        f"allocated={torch.cuda.memory_allocated()/1e9:.2f}GB"
                    )

                log("[APP] Loading MarianMT...")
                mt_loaded = load_local_mt(target_lang) if _TRANSFORMERS_OK else None
                tok, mt_model = (mt_loaded if mt_loaded else (None, None))

                if _TRANSFORMERS_OK and torch is not None and torch.cuda.is_available():
                    log(
                        f"[CUDA] after MarianMT reserved={torch.cuda.memory_reserved()/1e9:.2f}GB "
                        f"allocated={torch.cuda.memory_allocated()/1e9:.2f}GB"
                    )

            st.session_state.streaming_running = True
            st.session_state.stop_event.clear()

            log("[APP] Launching stream thread...")
            threading.Thread(
                target=stream_worker,
                args=(
                    updates_q,
                    st.session_state.stop_event,
                    whisper_model,
                    tok,
                    mt_model,
                    target_lang,
                    st.session_state.wav_path,
                    beam_size,
                    best_of,
                    float(chunk_sec),
                    float(overlap_sec),
                    bool(realtime_sleep),
                ),
                daemon=True,
            ).start()

        if st.session_state.streaming_running:
            st.info("⏳ Streaming in progress (updates every chunk).")
            if st.button("🛑 Stop streaming"):
                st.session_state.stop_event.set()
                st.session_state.streaming_running = False

    # ---------------- Gemini refinement (MAIN thread; quota-friendly) ----------------
    if st.session_state.gemini_enabled and _GEMINI_OK and st.session_state.streaming_running:
        now = time.time()
        chunk_ready = st.session_state.maybe_refine_due and (
            st.session_state.chunk_since_last_refine >= st.session_state.gemini_refine_every_chunks
        )
        block_cleared = now >= st.session_state.gemini_block_until_ts

        if (
            chunk_ready
            and block_cleared
            and os.environ.get("GEMINI_API_KEY", "").strip()
            and st.session_state.full_transcript.strip()
        ):
            log(
                f"[GEMINI] Refinement triggered "
                f"(chunk_ready={chunk_ready}, chunk_since_last_refine={st.session_state.chunk_since_last_refine}, "
                f"threshold={st.session_state.gemini_refine_every_chunks})"
            )

            t_tail = st.session_state.full_transcript[-GEMINI_CONTEXT_CHARS:]
            local_tail = st.session_state.full_translation_local[-GEMINI_TRANSLATION_CONTEXT_CHARS:]

            st.session_state.gemini_last_call_ts = now
            st.session_state.maybe_refine_due = False
            st.session_state.chunk_since_last_refine = 0

            try:
                refined = gemini_refine_translation(t_tail, local_tail, target_lang)
                if refined and not refined.startswith("(Gemini not configured"):
                    st.session_state.full_translation_refined = refined
                    st.session_state.gemini_last_error = ""
                    log(f"[GEMINI] Refined len={len(refined)}")
            except Exception as e:
                msg = str(e)
                st.session_state.gemini_last_error = msg
                log(f"[GEMINI] Error: {msg}")

                # crude backoff if quota/429
                if "429" in msg or "quota" in msg.lower():
                    st.session_state.gemini_block_until_ts = time.time() + 60

if __name__ == "__main__":
    app_ui()
