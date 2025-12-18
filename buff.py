import os
import re
import time
import uuid
import queue
import requests
import wave
import tempfile
import subprocess
import threading
import shutil
from pathlib import Path
from typing import Callable, Dict, Tuple, List, Optional

import streamlit as st
import numpy as np
from faster_whisper import WhisperModel


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


def log_gpu(tag: str) -> None:
    """Best-effort GPU snapshot using nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory",
        "--format=csv,nounits,noheader",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            for line in res.stdout.strip().splitlines():
                if line:
                    log(f"[GPU] {tag}: {line}")
        else:
            log(f"[GPU] {tag}: nvidia-smi exit {res.returncode} stderr={res.stderr.strip()}")
    except Exception as e:
        log(f"[GPU] {tag}: nvidia-smi failed: {e}")

# Optional: streamlit autorefresh
try:
    from streamlit_autorefresh import st_autorefresh
    _AUTOREFRESH_OK = True
except Exception:
    _AUTOREFRESH_OK = False

# Torch (for Whisper + CUDA checks)
try:
    import torch
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


# ---------------- CONFIG ----------------
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

DEFAULT_QWEN_REFINE_EVERY_CHUNKS = 3
QWEN_MODEL = "qwen2.5:7b-instruct"
QWEN_ENDPOINT = "http://localhost:11434/api/chat"

MT_MODEL = "zongwei/gemma3-translator:1b"
MT_ENDPOINT = "http://localhost:11434/api/chat"

CANONICAL_EN = (
    """Indeed, all praise is for Allah. We praise Him, seek His help, and seek His forgiveness.
We seek refuge in Allah from the evil within ourselves and from the evil of our deeds.
Whomever Allah guides, none can mislead; and whomever He leaves astray, none can guide.
I bear witness that there is no deity worthy of worship except Allah alone, with no partner;
and I bear witness that Muhammad is His servant and His Messenger."""
)

CANONICAL_DE = (
    """Wahrlich, alles Lob gebührt Allah. Wir lobpreisen Ihn, suchen Seine Hilfe und bitten Ihn um Vergebung.
Wir suchen Zuflucht bei Allah vor dem Bösen in unseren Seelen und vor den bösen Folgen unserer Taten.
Wen Allah rechtleitet, den kann niemand in die Irre führen; und wen Er in die Irre gehen lässt, den kann niemand rechtleiten.
Ich bezeuge, dass es keine Gottheit gibt, die anbetungswürdig ist, außer Allah allein, ohne Teilhaber;
und ich bezeuge, dass Muhammad Sein Diener und Sein Gesandter ist."""
)

INITIAL_PROMPT_ISLAMIC = (
    "This is an Arabic Islamic lecture. Preserve Islamic terms faithfully: Allah, Qur'an, Sunnah, hadith, salah, zakah, sawm, Hajj, Umrah, tawhid."
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

# Stable placeholder format unlikely to be mutated by MT
PH_FMT = "§§TERM{n}§§"
TERM_RE = re.compile(r"(?:<<\s*TERM_?\s*(\d+)\s*>>|<\s*TERM_?\s*(\d+)\s*>|>>\s*TERM_?\s*(\d+)\s*>>|§§TERM(\d+)§§)")
MT_META_RE = re.compile(
    r"(?i)\b(here'?s|note:|translation:|i aimed|section|explanation|summary)\b|```",
    re.M,
)
QWEN_META_RE = re.compile(r"(?i)\b(note:|here'?s|translation:|section)\b|```", re.M)
MAX_MT_CHARS = 700
QWEN_AR_TAIL = 2200
QWEN_DRAFT_TAIL = 2200

# Arabic normalization for khutbah detection
_AR_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_AR_NONLETTER = re.compile(r"[^\u0600-\u06FF\s]+")

def _norm_ar(s: str) -> str:
    s = _AR_DIACRITICS.sub("", s)
    s = _AR_NONLETTER.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Khutbah al-Hajah detection markers
KHUTBAH_FULL_MARKERS = [
    "الحمد لله نحمده",
    "ونستعينه",
    "ونستغفره",
    "ونعوذ بالله",
    "من يهده الله",
    "فلا مضل له",
    "ومن يضلل",
    "وأشهد أن لا إله إلا الله",
    "وأشهد أن محمدا",
]

KHUTBAH_ANY_MARKERS = [
    "الحمد لله نحمده",
    "من يهده الله",
    "وأشهد أن لا إله إلا الله",
    "وأشهد أن محمدا",
]

def khutbah_opening_status(ar: str) -> Tuple[bool, bool]:
    """Returns (is_full, is_fragment) - full means complete opening, fragment means partial."""
    n = _norm_ar(ar)
    full = all(m in n for m in KHUTBAH_FULL_MARKERS)
    fragment = (not full) and any(m in n for m in KHUTBAH_ANY_MARKERS)
    return full, fragment

def canonical_for(lang: str) -> str:
    return CANONICAL_EN if lang == "English" else CANONICAL_DE


def protect_terms(text: str, target_lang: str) -> Tuple[str, Dict[str, str]]:
    placeholder_map: Dict[str, str] = {}
    out = text
    keys = sorted(GLOSSARY.keys(), key=len, reverse=True)
    idx = 0
    for k in keys:
        if k in out:
            idx += 1
            ph = PH_FMT.format(n=idx)
            placeholder_map[ph] = GLOSSARY[k].get(target_lang, k)
            out = out.replace(k, ph)
    return out, placeholder_map


def restore_terms(text: str, placeholder_map: Dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        n = next(g for g in m.groups() if g is not None)
        return PH_FMT.format(n=n)

    normalized = TERM_RE.sub(repl, text)

    out = normalized
    for ph, val in placeholder_map.items():
        out = out.replace(ph, val)
    return out


def validate_mt_output(text: str) -> bool:
    if not text.strip():
        return False
    if MT_META_RE.search(text):
        return False
    # Must not corrupt placeholders
    if "§§TERM" in text and not re.search(r"§§TERM\d+§§", text):
        return False
    return True


def force_chunk(text: str, max_chars: int) -> List[str]:
    words = text.split()
    out, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > max_chars and cur:
            out.append(cur.strip())
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        out.append(cur.strip())
    return out


# ---------------- Per-session queue (stable across reruns) ----------------
def get_updates_q() -> "queue.Queue[Tuple[str, dict]]":
    if "updates_q" not in st.session_state:
        st.session_state.updates_q = queue.Queue()
    return st.session_state.updates_q


# ---------------- Whisper / MT loaders (MAIN THREAD ONLY) ----------------
@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    cuda_available = False
    try:
        import torch  # type: ignore
        cuda_available = torch.cuda.is_available()
    except Exception:
        pass
    log(
        f"[WHISPER] Loading model={model_size} device={device} compute_type={compute_type} cuda_available={cuda_available}"
    )
    # local-only first, then allow download
    try:
        mdl = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=WHISPER_CACHE_DIR,
            local_files_only=True,
        )
        log(f"[WHISPER] Loaded {model_size} on device={device} (local cache)")
        return mdl
    except Exception:
        log("[WHISPER] local-only load failed, retrying with download allowed")
        mdl = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=WHISPER_CACHE_DIR,
            local_files_only=False,
        )
        log(f"[WHISPER] Loaded {model_size} on device={device} (download ok)")
        return mdl


# Ollama MT (GPU forced)
def build_mt_system(lang: str) -> str:
    if lang not in ("English", "German"):
        raise ValueError("target_lang must be 'English' or 'German'")

    if lang == "English":
        lang_rules = """- Target language: English.
- Use "Allah" for "الله" (never "God").
- Translate "لا إله إلا الله" as: "There is no deity worthy of worship except Allah."
- Translate "أما بعد" as "To proceed:"
- Keep a formal khutbah/lecture tone."""
    else:
        lang_rules = """- Zielsprache: Deutsch.
- Verwende "Allah" für "الله" (niemals "Gott").
- Übersetze "لا إله إلا الله" als: "Es gibt keine Gottheit, die anbetungswürdig ist, außer Allah."
- Übersetze "أما بعد" als "Sodann:"
- Bewahre einen formellen Khutbah-/Vortragston."""

    return f"""You are a strict Arabic→{lang} translator for Islamic khutbahs.

ABSOLUTE RULES:
- Output ONLY the translation. No preface, no notes, no headings, no quotes, no bullet points.
- Do NOT add, remove, paraphrase, summarize, merge, or reorder information.
- If the Arabic is unclear or contains ASR errors, translate it literally as written; do not invent meaning.
- NEVER expand a partial phrase into a longer known text (even if it resembles a famous opening).
- Preserve all placeholders EXACTLY (do not change them): any token like §§TERM123§§ must appear unchanged.
- Preserve Qur'anic/hadith quoting tone. Do not modernize religious phrasing.

RELIGIOUS CONSISTENCY:
{lang_rules}
"""


def ollama_translate_sentence(text_ar: str, target_lang: str) -> str:
    if not text_ar.strip():
        return ""

    # Check if this is the full khutbah opening - return canonical directly
    is_full, is_fragment = khutbah_opening_status(text_ar)
    if is_full:
        return canonical_for(target_lang).strip()

    protected, ph_map = protect_terms(text_ar, target_lang)
    system = build_mt_system(target_lang)

    # Try up to 3 times with increasingly strict reminders
    for attempt in range(3):
        payload = {
            "model": MT_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": protected},
            ],
            "stream": False,
            "options": {
                "temperature": 0,
                "top_p": 0.1,
                "num_ctx": 2048,
                "num_gpu": 1,
                "repeat_penalty": 1.15,
                "num_predict": 380,  # prevent explosion into long text
                "stop": ["```", "\nNote:", "\nNOTE:", "\nHere", "\nTranslation:", "\nSECTION"],
            },
        }

        resp = requests.post(MT_ENDPOINT, json=payload, timeout=60)
        resp.raise_for_status()
        translated = ((resp.json().get("message") or {}).get("content") or "").strip()
        translated = restore_terms(translated, ph_map)

        # If fragment and model hallucinated the full canonical, reject it
        if is_fragment and "Indeed, all praise is for Allah" in translated:
            translated = ""

        if validate_mt_output(translated):
            return translated

        # tighten system on retry
        system += "\n\nFINAL WARNING: Translate ONLY the given Arabic fragment. NEVER output a longer known text."

    # best-effort fallback
    return translated


# ---------------- Qwen via Ollama (MAIN THREAD ONLY) ----------------
def build_qwen_polish_system(lang: str) -> str:
    if lang not in ("English", "German"):
        raise ValueError("target_lang must be 'English' or 'German'")

    if lang == "English":
        lang_rules = '- Use "Allah" (never "God").'
    else:
        lang_rules = '- Use "Allah" (never "Gott").'

    return f"""You are polishing a draft {lang} translation of an Arabic Islamic khutbah.

ABSOLUTE RULES:
- Output ONLY the polished {lang} text. No preface, no notes, no headings, no quotes.
- Do NOT add, remove, reorder, or summarize meaning.
- Correct mistranslations and improve fluency while staying faithful.
{lang_rules}
- Preserve Islamic terms: Allah, Qur'an, Sunnah, hadith, salah, zakah, sawm, Hajj, Umrah, tawhid.
"""


def validate_qwen(out: str) -> bool:
    return bool(out.strip()) and not QWEN_META_RE.search(out)


def qwen_refine_translation(ar_tail: str, draft_tail: str, target_lang: str, model_name: str = QWEN_MODEL) -> str:
    system = build_qwen_polish_system(target_lang)
    user_content = f"ARABIC (context):\n{ar_tail}\n\nDRAFT ({target_lang}):\n{draft_tail}\n\nReturn ONLY the corrected draft."

    # Try twice with stricter system on retry
    for attempt in range(2):
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "options": {
                "temperature": 0,
                "top_p": 0.1,
                "num_ctx": 4096,
                "num_gpu": 1,  # use GPU
            },
        }

        try:
            log(
                f"[QWEN] Sending polish request ar_len={len(ar_tail)} draft_len={len(draft_tail)} model={model_name} attempt={attempt+1}"
            )
            resp = requests.post(QWEN_ENDPOINT, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("message", {}) or {}).get("content", "").strip()
            
            if validate_qwen(text):
                return text
            
            log(f"[QWEN] Output failed validation on attempt {attempt+1}")
            # Tighten system for retry
            system = system + "\n\nFINAL WARNING: Output ONLY the polished translation text. NO meta commentary."
        except Exception as e:
            log(f"[QWEN] Error during refine attempt {attempt+1}: {e}")
            if attempt == 1:
                raise
    
    return text  # best-effort fallback


# ---------------- Audio extraction ----------------
def video_to_wav(video_path: str, wav_path: str, sr: int = 16000) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
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


def split_complete_sentences(s: str) -> Tuple[List[str], str]:
    """Split text into completed sentences and remainder using simple punctuation heuristics."""
    parts = re.split(r"([\.\!\?؟\n]+)", s)
    out: List[str] = []
    cur = ""
    for i in range(0, len(parts), 2):
        chunk = parts[i]
        end = parts[i + 1] if i + 1 < len(parts) else ""
        cur += chunk + end
        if end:
            out.append(cur.strip())
            cur = ""
    return out, cur


def transcribe_file_once_with_model(model: WhisperModel, path: str, beam_size: int, best_of: int, initial_prompt: str) -> str:
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
    target_lang: str,
    wav_path: str,
    beam_size: int,
    best_of: int,
    chunk_sec: float,
    overlap_sec: float,
    realtime_sleep: bool,
):
    log(f"[STREAM] start qid={id(updates_q)}")
    log_gpu("stream start")
    full_text = ""
    rolling_prompt = INITIAL_PROMPT_ISLAMIC
    ar_buf = ""
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

                    new_ar = delta
                    if new_ar:
                        ar_buf = (ar_buf + " " + new_ar).strip()

                    local_delta_lines: List[str] = []
                    done_sents: List[str] = []
                    if ar_buf:
                        done_sents, ar_buf = split_complete_sentences(ar_buf)

                    # If no sentence boundary yet but buffer is long, translate to avoid UI stalling
                    if not done_sents and len(ar_buf) > 120:
                        done_sents = [ar_buf]
                        ar_buf = ""

                    if done_sents:
                        for sent in done_sents:
                            for piece in force_chunk(sent, MAX_MT_CHARS):
                                try:
                                    translated = ollama_translate_sentence(piece, target_lang)
                                except Exception as e:
                                    log(f"[MT] Translation error: {e}")
                                    translated = ""
                                if translated:
                                    local_delta_lines.append(translated)

                    local_delta = "\n".join(local_delta_lines).strip()

                    updates_q.put(("chunk", {
                        "full_transcript": full_text,
                        "delta_ar": delta,
                        "local_delta": local_delta,
                        "chunk_idx": chunk_count,
                    }))
                    chunk_count += 1
                    log(
                        f"[STREAM] Chunk {chunk_count}: delta_ar_len={len(delta)}, local_delta_len={len(local_delta)}\n"
                        f"    delta_ar='{delta[:160]}'\n    local_delta='{local_delta[:160]}'"
                    )
                    log_gpu(f"after chunk {chunk_count}")

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

    # Qwen refine control (main thread only)
    if "qwen_enabled" not in st.session_state:
        st.session_state.qwen_enabled = False
    if "qwen_last_error" not in st.session_state:
        st.session_state.qwen_last_error = ""
    if "qwen_refine_every_chunks" not in st.session_state:
        st.session_state.qwen_refine_every_chunks = DEFAULT_QWEN_REFINE_EVERY_CHUNKS
    if "qwen_force_refine" not in st.session_state:
        st.session_state.qwen_force_refine = False
    if "qwen_was_enabled" not in st.session_state:
        st.session_state.qwen_was_enabled = False
    if "qwen_inflight" not in st.session_state:
        st.session_state.qwen_inflight = False
    if "qwen_last_call_ts" not in st.session_state:
        st.session_state.qwen_last_call_ts = 0.0

    if "start_stream_request" not in st.session_state:
        st.session_state.start_stream_request = False

    if "chunk_total" not in st.session_state:
        st.session_state.chunk_total = 0

    # log file paths (per-session temp files)
    if "transcript_log_path" not in st.session_state:
        base_dir = Path(__file__).resolve().parent
        st.session_state.transcript_log_path = str(base_dir / f"transcript_{st.session_state.session_id}.txt")
    if "translation_log_path" not in st.session_state:
        base_dir = Path(__file__).resolve().parent
        st.session_state.translation_log_path = str(base_dir / f"translation_{st.session_state.session_id}.txt")
    ensure_log_file(st.session_state.transcript_log_path)
    ensure_log_file(st.session_state.translation_log_path)

    # UI-bound text widgets (kept in sync each rerun)
    if "transcript_display" not in st.session_state:
        st.session_state.transcript_display = ""
    if "translation_display" not in st.session_state:
        st.session_state.translation_display = ""


def drain_updates(updates_q: "queue.Queue[Tuple[str, dict]]") -> bool:
    changed = False
    while True:
        try:
            kind, payload = updates_q.get_nowait()
        except queue.Empty:
            break

        changed = True

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
                append_line(st.session_state.transcript_log_path, "\n---- FULL TRANSCRIPT @90 ----\n" + st.session_state.full_transcript + "\n---- END ----\n")
                append_line(st.session_state.translation_log_path, "\n---- FULL TRANSLATION @90 ----\n" + st.session_state.full_translation_local + "\n---- END ----\n")

            st.session_state.chunk_since_last_refine += 1
            if st.session_state.chunk_since_last_refine >= st.session_state.qwen_refine_every_chunks:
                st.session_state.maybe_refine_due = True
                log(f"[DRAIN] Qwen refine due (chunk #{st.session_state.chunk_since_last_refine})")

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
    changed = drain_updates(updates_q)
    # Sync widget-bound values so st.text_area reflects latest text each rerun
    st.session_state.transcript_display = st.session_state.full_transcript
    st.session_state.translation_display = st.session_state.full_translation_local


    # Refresh frequently to pick up background queue updates (always on per-session)
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
        st.subheader("Qwen refinement (optional)")
        st.caption("Runs locally via Ollama. Uses the full transcript every N chunks.")
        st.session_state.qwen_enabled = st.checkbox("Enable Qwen refinement", value=st.session_state.qwen_enabled)
        st.session_state.qwen_refine_every_chunks = st.slider(
            "Refine every N chunks", 1, 50, st.session_state.qwen_refine_every_chunks, 1
        )
        if st.button("Refine now (Qwen)", disabled=not st.session_state.qwen_enabled):
            st.session_state.qwen_force_refine = True
            st.session_state.maybe_refine_due = True

        if st.session_state.qwen_enabled and not st.session_state.qwen_was_enabled:
            # Fire a refine soon after enabling if transcript exists
            st.session_state.maybe_refine_due = True
        st.session_state.qwen_was_enabled = st.session_state.qwen_enabled

        st.divider()
        if st.button("Warm-load Whisper"):
            with st.spinner("Loading Whisper..."):
                _ = load_whisper_model(model_size, device, compute_type)
            st.success("Whisper loaded & cached.")

    # Main panels
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transcription (Arabic)")
        st.text(" ")  # spacer to keep layout stable
        st.empty().markdown(f"``\n{st.session_state.transcript_display}\n``")

    with col2:
        st.subheader(f"Translation (Local, {target_lang})")
        st.text(" ")
        st.empty().markdown(f"``\n{st.session_state.translation_display}\n``")

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
        f"extraction_done={st.session_state.extraction_done}"
    )

    st.subheader("Refined Translation (Qwen via Ollama)")
    st.text(" ")
    st.empty().markdown(f"``\n{st.session_state.full_translation_refined}\n``")
    if st.session_state.qwen_last_error:
        st.warning(f"Qwen error: {st.session_state.qwen_last_error[:260]}...")

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
            st.session_state.qwen_last_error = ""
            st.session_state.last_delta_ar = ""
            st.session_state.last_delta_local = ""
            st.session_state.chunk_since_last_refine = 0
            st.session_state.maybe_refine_due = False

            st.session_state.streaming_running = False
            st.session_state.stop_event.set()   # stop any prior worker
            st.session_state.stop_event = threading.Event()  # new clean event

            st.session_state.extraction_done = False
            st.session_state.extraction_in_progress = False
            st.session_state.extraction_error = ""

            # save video
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
                tmp.write(uploaded.getbuffer())
                st.session_state.video_path = tmp.name

            st.session_state.wav_path = os.path.join(tempfile.gettempdir(), f"extracted_{time.time_ns()}.wav")
            log(f"[APP] New upload saved: video={st.session_state.video_path}, wav={st.session_state.wav_path}")

        st.video(st.session_state.video_path)

        # Start extraction ONCE
        if (not st.session_state.extraction_done) and (not st.session_state.extraction_in_progress) and (not st.session_state.extraction_error):
            st.session_state.extraction_in_progress = True
            log("[APP] Starting extraction thread")
            threading.Thread(
                target=extraction_worker,
                args=(updates_q, st.session_state.video_path, st.session_state.wav_path),
                daemon=True,
            ).start()
            # Trigger a single rerun so autorefresh and drain will see extraction_in_progress
            st.rerun()

        # Status
        if st.session_state.extraction_in_progress:
            st.info("🎧 Extracting audio with ffmpeg...")

        if st.session_state.extraction_error:
            st.error(f"⚠️ Extraction failed: {st.session_state.extraction_error}")

        # Start streaming button appears deterministically once extraction_done flips true
        if st.session_state.extraction_done and (not st.session_state.streaming_running):
            st.success("✅ Audio extracted. Ready to stream.")
            start_clicked = st.button("🎬 Start streaming transcription", key="start_stream_btn")
            if start_clicked:
                st.session_state.start_stream_request = True
                log("[APP] Start streaming requested")

        # honor start request even if the button click was followed by an autorefresh rerun
        if st.session_state.start_stream_request and st.session_state.extraction_done and (not st.session_state.streaming_running):
            st.session_state.start_stream_request = False
            log("[APP] Starting streaming worker now")

            if torch.cuda.is_available():
                log(
                    f"[CUDA] pre-load reserved={torch.cuda.memory_reserved()/1e9:.2f}GB allocated={torch.cuda.memory_allocated()/1e9:.2f}GB"
                )

            # preload models in MAIN thread (critical!)
            with st.spinner("Loading models (one-time)..."):
                log("[APP] Loading Whisper...")
                whisper_model = load_whisper_model(model_size, device, compute_type)
                log("[APP] Whisper loaded.")
                log_gpu("after Whisper load")
                if torch.cuda.is_available():
                    log(
                        f"[CUDA] after Whisper reserved={torch.cuda.memory_reserved()/1e9:.2f}GB allocated={torch.cuda.memory_allocated()/1e9:.2f}GB"
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

    # ---------------- Qwen refinement (MAIN thread; local Ollama) ----------------
    if st.session_state.qwen_enabled:
        chunk_ready = st.session_state.maybe_refine_due and (
            st.session_state.chunk_since_last_refine >= st.session_state.qwen_refine_every_chunks
        )
        force_ready = st.session_state.qwen_force_refine
        
        if (chunk_ready or force_ready) and st.session_state.full_transcript.strip():
            now = time.time()
            if st.session_state.qwen_inflight:
                pass  # Silently skip inflight (normal during long Qwen calls)
            elif now - st.session_state.qwen_last_call_ts < 5:  # 5s throttle
                pass  # Silently skip, this is normal during autorefresh
            else:
                reason = "manual" if force_ready else "auto (every N chunks)"
                log(f"[QWEN] Starting refinement ({reason})")

                st.session_state.qwen_inflight = True
                st.session_state.qwen_last_call_ts = now
                st.session_state.maybe_refine_due = False
                st.session_state.chunk_since_last_refine = 0
                st.session_state.qwen_force_refine = False
                
                try:
                    # Use tail windows instead of full transcript
                    ar_tail = st.session_state.full_transcript[-QWEN_AR_TAIL:]
                    draft_tail = st.session_state.full_translation_local[-QWEN_DRAFT_TAIL:]
                    
                    refined = qwen_refine_translation(ar_tail, draft_tail, target_lang)
                    if refined:
                        st.session_state.full_translation_refined = refined
                        st.session_state.qwen_last_error = ""
                        log(f"[QWEN] Refined len={len(refined)}\n    qwen_output='{refined[:200]}'")
                except Exception as e:
                    msg = str(e)
                    st.session_state.qwen_last_error = msg
                    log(f"[QWEN] Error: {msg}")
                finally:
                    st.session_state.qwen_inflight = False


if __name__ == "__main__":
    app_ui()
