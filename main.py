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
import gc
import datetime
import difflib
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from streamlit.runtime.scriptrunner import add_script_run_ctx

# ==========================================
# 1. CONFIGURATION & PROMPTS
# ==========================================

OLLAMA_API_URL = "http://localhost:11434/api/chat"
DEFAULT_MT_MODEL = "qwen2.5:3b-instruct"

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
5) If the input is not Arabic (e.g., German/English), output exactly: [non-arabic]."""
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


def robust_ollama_call(payload, timeout=30, retries=1):
    for attempt in range(retries + 1):
        try:
            resp = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            text = resp.json().get("message", {}).get("content", "").strip()
            
            # --- CLEANER ---
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text = re.sub(r'^(Here is|Sure|Translation|Corrected|Okay|I have translated).*?:', '', text, flags=re.IGNORECASE).strip()
            text = re.sub(r'(Note:|P.S.|Analysis:).*', '', text, flags=re.IGNORECASE).strip()
            text = text.strip('"').strip("'")
            # ---------------

            if re.search(r'[\u4e00-\u9fff]', text): return None 
            return text
        except Exception as e:
            log(f"[API] Error: {e}")
        if attempt < retries: time.sleep(1)
    return None

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
def refinement_worker(input_q, output_q, config=None):
    log("Refinement Worker: STARTING")

    # Default prompts (will be overridden per-job if 'prompts' passed)
    current_fixer_prompt = PROMPT_TEMPLATES["Standard (Fusha/MSA)"]["fixer"]
    current_translator_prompt = PROMPT_TEMPLATES["Standard (Fusha/MSA)"]["translator"]

    # Conservative guard thresholds (tune if needed)
    SIMILARITY_MIN = 0.88          # reject fixer output if it rewrites too much
    MIN_AR_RATIO = 0.65            # reject if "Arabic-ness" drops too far

    def sanitize_fixed_ar(raw_ar: str, fixed_ar: str) -> str:
        """Reject fixer output if it rewrites, adds English, or drifts away from Arabic."""
        if not fixed_ar:
            return raw_ar

        fx = fixed_ar.strip()

        # Reject if it inserted Latin letters (we want Arabic-only for fixed_ar)
        if re.search(r"[A-Za-z]", fx):
            return raw_ar

        # Reject if it looks like non-arabic drift
        if arabic_ratio(fx) < MIN_AR_RATIO:
            return raw_ar

        # Reject if it rewrote too much (hallucination / paraphrase)
        if similarity(raw_ar, fx) < SIMILARITY_MIN:
            return raw_ar

        return fx

    while True:
        try:
            job = input_q.get(timeout=2)
            if job is None:
                continue
            if job == "STOP":
                break

            raw_ar = (job.get("source_ar") or "").strip()
            batch_id = job.get("id", 0)
            log_file_ar = job.get("log_file_ar", None)
            log_file_en = job.get("log_file_en", None)
            t_submitted = job.get("ts", time.time())

            if not raw_ar:
                continue

            # Allow per-job style prompt overrides
            if "prompts" in job and job["prompts"]:
                current_fixer_prompt = job["prompts"].get("fixer", current_fixer_prompt)
                current_translator_prompt = job["prompts"].get("translator", current_translator_prompt)

            # If the speaker switched language (not Arabic), skip "fixing" and translate as [non-arabic]
            if arabic_ratio(raw_ar) < 0.50:
                corrected_ar = raw_ar
                final_en = "[non-arabic]"
                output_q.put({
                    "type": "refined_batch",
                    "id": batch_id,
                    "ar_fixed": corrected_ar,
                    "en_final": final_en
                })
                continue

            log(f"Refining Batch {batch_id}...")

            # -------------------------
            # 1) Conservative Restorer
            # -------------------------
            # IMPORTANT: keep this deterministic and conservative.
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
                    {"role": "user", "content": raw_ar}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 0.2,
                    "num_ctx": 1400
                }
            }, timeout=50)

            corrected_ar = sanitize_fixed_ar(raw_ar, fixed_candidate)

            lat_fix = time.time() - t_submitted
            if log_file_ar:
                append_to_file(
                    log_file_ar,
                    f"[{get_current_ts_string()}] [Batch {batch_id}] (Lat: {lat_fix:.2f}s)\n{corrected_ar}\n"
                )

            # -------------------------
            # 2) Refined Translation
            # -------------------------
            # Make translation literal + glossary-bound; forbid additions.
            translator_sys = (
                current_translator_prompt.strip()
                + "\n\n"
                + glossary_block()
                + "\n\nCRITICAL RULES (must follow):\n"
                  "- Translate ONLY what is present in the Arabic input.\n"
                  "- Do NOT add khutbah phrases, Qur’anic verses, or explanations that are not in the text.\n"
                  "- If a phrase is garbled/unclear, write [unclear] rather than guessing.\n"
                  "- Output ONLY the translation.\n"
            )

            final_en = robust_ollama_call({
                "model": DEFAULT_MT_MODEL,
                "messages": [
                    {"role": "system", "content": translator_sys},
                    {"role": "user", "content": corrected_ar}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 0.2,
                    "num_ctx": 1800
                }
            }, timeout=50)

            # Fallback if translation fails
            if not final_en:
                final_en = "[unclear]"

            lat_total = time.time() - t_submitted

            output_q.put({
                "type": "refined_batch",
                "id": batch_id,
                "ar_fixed": corrected_ar,
                "en_final": final_en
            })

            if log_file_en:
                append_to_file(
                    log_file_en,
                    f"[{get_current_ts_string()}] [Batch {batch_id}] (Total Lat: {lat_total:.2f}s)\n{final_en}\n"
                )

            log(f"Batch {batch_id} Done (Total Lat: {lat_total:.2f}s).")

        except queue.Empty:
            continue
        except Exception as e:
            log(f"Refinement Worker Crash: {e}")
            time.sleep(1)


# --- WORKER 2: INSTANT TRANSLATION ---
def instant_translation_worker(transcription_q, event_q, config):
    log("Instant Translator: STARTING")
    base_translator_prompt = config['prompts']['translator']
    log_draft_en = config['logs']['draft_en']

    # Strong anti-hallucination translation rules + glossary (no placeholders!)
    sys_prompt = (
        base_translator_prompt.strip()
        + "\n\n"
        + glossary_block()
        + "\n\nCRITICAL RULES (must follow):\n"
          "- Translate ONLY what is present in the Arabic input. Do NOT add missing khutbah phrases.\n"
          "- If a phrase is garbled/unclear, write [unclear] for that part rather than guessing.\n"
          "- If the input is not Arabic (e.g., German/English), output exactly: [non-arabic]\n"
          "- Keep proper Islamic terms according to the glossary.\n"
          "- Output ONLY the translation text.\n"
    )

    while True:
        try:
            item = transcription_q.get(timeout=1)
            if item is None:
                break

            chunk_id = item['id']
            ar_text = (item.get('text') or "").strip()
            t_emit = item.get('ts', time.time())

            # If it’s not Arabic (speaker switched language), skip LLM call.
            if arabic_ratio(ar_text) < 0.50:
                final_tr = "[non-arabic]"
            else:
                raw_tr = robust_ollama_call({
                    "model": config['mt_model'],
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": ar_text}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.0,   # deterministic -> less “sermon creativity”
                        "top_p": 0.2,
                        "num_ctx": 1024
                    }
                }, timeout=45) or "[unclear]"

                final_tr = raw_tr.strip()

            latency = time.time() - t_emit
            append_to_file(log_draft_en, f"[{get_current_ts_string()}] [{chunk_id}] (Lat: {latency:.2f}s) {final_tr}")

            event_q.put(("update", {"id": chunk_id, "ar": ar_text, "tr": final_tr}))

        except queue.Empty:
            continue
        except Exception as e:
            log(f"Translator Error: {e}")


# --- WORKER 3: AUDIO & WHISPER (Combined Fixes) ---
def transcription_stream_thread(source, config, trans_q, stop_event, refine_input_q, event_q):
    """
    SimulStreaming-core replication (practical version for faster-whisper):
      - VAC/VAD iterator: discard non-voice, accumulate voiced chunks, mark end-of-voice as final.
      - LocalAgreement: commit only the longest common prefix between consecutive hypotheses.
      - Boundary safety: trim last word on non-final chunks (prevents partial-word garbage).
      - Sliding audio buffer (max 30s) + reset on end-of-voice.
    """
    import time
    import wave
    import re
    import datetime
    from collections import deque
    import numpy as np

    # -----------------------------
    # 0) Helpers
    # -----------------------------
    def get_ts():
        return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

    log_raw = config["logs"]["raw_ar"]

    def log_to_file(text):
        try:
            with open(log_raw, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except:
            pass

    # Normalize Arabic-ish whitespace (don’t overdo it)
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
        """Longest common prefix at word level."""
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return a[:i]

    def trim_last_word(words):
        """Drop last word (chunk boundary safety)"""
        if not words:
            return words
        return words[:-1]

    # -----------------------------
    # 1) Rolling audio buffer (same idea you had)
    # -----------------------------
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

    # -----------------------------
    # 2) VAC / VAD iterator (SimulStreaming-style)
    #    Paper describes 0.04s frames + min_nonvoice 500ms + padding 100ms, and
    #    hold voice until MinChunkSize seconds or end-of-voice. :contentReference[oaicite:8]{index=8}
    # -----------------------------
    try:
        import webrtcvad
        _VAD_OK = True
    except Exception:
        _VAD_OK = False

    class VacIterator:
        """
        Yields (voiced_audio_np_float32, is_final)
        - Reads fixed frames (frame_sec)
        - Discards non-voice frames
        - Accumulates voiced frames until:
            * voiced >= min_voiced_sec -> yield (is_final=False)
            * or end-of-voice detected (silence >= min_silence_ms) -> yield remainder (is_final=True)
        """
        def __init__(self, sr=16000, frame_sec=0.04, min_silence_ms=500, pad_ms=100, min_voiced_sec=1.0, vad_mode=2):
            self.sr = int(sr)
            self.frame_sec = float(frame_sec)
            self.frame_samples = int(self.sr * self.frame_sec)
            self.min_silence_ms = int(min_silence_ms)
            self.pad_samples = int(self.sr * (pad_ms / 1000.0))
            self.min_voiced_samples = int(self.sr * float(min_voiced_sec))

            self.vad = webrtcvad.Vad(vad_mode) if _VAD_OK else None
            self.in_voice = False
            self.silence_ms = 0

            self.pad_buf = deque()          # holds last pad_samples of audio
            self.voiced_chunks = []         # list of np arrays for current voiced accumulation
            self.voiced_samples = 0

        def _is_speech(self, frame_f32: np.ndarray) -> bool:
            if self.vad is None:
                # Fallback energy gate (worse than real VAD but better than nothing)
                return float(np.sqrt(np.mean(frame_f32 * frame_f32))) > 0.01

            # WebRTC VAD expects 16-bit PCM mono
            pcm16 = np.clip(frame_f32 * 32768.0, -32768, 32767).astype(np.int16).tobytes()
            return self.vad.is_speech(pcm16, self.sr)

        def push_frame(self, frame_f32: np.ndarray):
            out = []

            if frame_f32 is None or len(frame_f32) == 0:
                return out

            # Keep rolling padding buffer
            self.pad_buf.append(frame_f32)
            # trim pad_buf to pad_samples total
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
                    # enter voice: prepend padding
                    self.in_voice = True
                    if self.pad_buf:
                        pad = np.concatenate(list(self.pad_buf))
                        self.voiced_chunks.append(pad)
                        self.voiced_samples += len(pad)
                    self.silence_ms = 0

                self.voiced_chunks.append(frame_f32)
                self.voiced_samples += len(frame_f32)
                self.silence_ms = 0

                # If enough voiced audio accumulated, emit non-final chunk and keep going
                if self.voiced_samples >= self.min_voiced_samples:
                    chunk = np.concatenate(self.voiced_chunks) if self.voiced_chunks else np.zeros((0,), np.float32)
                    if len(chunk) > 0:
                        out.append((chunk, False))
                    self.voiced_chunks = []
                    self.voiced_samples = 0
                return out

            # non-speech
            if self.in_voice:
                self.silence_ms += int(self.frame_sec * 1000)

                # allow a little trailing padding by including a tiny bit of silence frames if you want
                # (optional). We'll keep it simple: don't append silence frames.

                if self.silence_ms >= self.min_silence_ms:
                    # end-of-voice => flush remainder as final
                    chunk = np.concatenate(self.voiced_chunks) if self.voiced_chunks else np.zeros((0,), np.float32)
                    if len(chunk) > 0:
                        out.append((chunk, True))
                    # reset state
                    self.in_voice = False
                    self.silence_ms = 0
                    self.voiced_chunks = []
                    self.voiced_samples = 0

            # discard non-voice frames entirely
            return out

    # -----------------------------
    # 3) Model load (faster-whisper or your loader)
    # -----------------------------
    try:
        model = load_whisper_model(config["model_size"], config["device"], config["compute_type"])
    except NameError:
        from faster_whisper import WhisperModel
        model = WhisperModel(config["model_size"], device=config["device"], compute_type=config["compute_type"])

    if not model:
        event_q.put(("error", "Model failed to load."))
        return

    # -----------------------------
    # 4) Audio source
    # -----------------------------
    import sounddevice as sd
    is_live_mic = isinstance(source, int)
    wf, stream = None, None
    sr = 16000

    if is_live_mic:
        try:
            # read in VAD frame sizes to align with VAC
            frame_sec = 0.04
            stream = sd.InputStream(device=source, channels=1, samplerate=sr, dtype="float32",
                                    blocksize=int(sr * frame_sec))
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

    # -----------------------------
    # 5) Filters / tuning
    # -----------------------------
    hallucinations_re = re.compile(
        r"(Subtitle|Translated|Amara|MBC|Copyright|Rights|Reserved|Music|"
        r"نانسي|ترجمة|اشتركوا|اشترك|الحقوق|حقوق|محفوظة|قناة|تابعونا|"
        r"شكرا|شكراً)",
        re.IGNORECASE,
    )

    # VAC parameters (matching paper defaults closely) :contentReference[oaicite:9]{index=9}
    VAD_FRAME_SEC = 0.04
    MIN_NONVOICE_MS = 500
    VOICE_PAD_MS = 100
    MIN_VOICED_ACCUM_SEC = float(config.get("min_voiced_accum_sec", 1.0))

    # Sliding audio buffer (like SimulStreaming buffer length) :contentReference[oaicite:10]{index=10}
    MAX_WINDOW_SEC = float(config.get("max_window_sec", 30.0))
    AUDIO_MIN_SEC = float(config.get("audio_min_sec", 1.0))

    audio_buf = RollingAudioBuffer(sr=sr, max_sec=MAX_WINDOW_SEC)
    total_samples_read = 0

    vac = VacIterator(
        sr=sr,
        frame_sec=VAD_FRAME_SEC,
        min_silence_ms=MIN_NONVOICE_MS,
        pad_ms=VOICE_PAD_MS,
        min_voiced_sec=MIN_VOICED_ACCUM_SEC,
        vad_mode=int(config.get("vad_mode", 2)),
    )

    # LocalAgreement state :contentReference[oaicite:11]{index=11}
    committed_words = []
    prev_hyp_words = []

    chunk_counter = 0
    accumulated_text = ""
    last_send_time = time.time()

    # -----------------------------
    # 6) Main loop
    # -----------------------------
    while not stop_event.is_set():
        # ---- read one VAD frame
        if is_live_mic:
            if not stream.active:
                break
            data, overflow = stream.read(int(sr * VAD_FRAME_SEC))
            frame = data.flatten().astype(np.float32)
        else:
            raw_bytes = wf.readframes(int(sr * VAD_FRAME_SEC))
            if not raw_bytes:
                break
            frame = np.frombuffer(raw_bytes, np.int16).astype(np.float32) / 32768.0

        if len(frame) == 0:
            break

        t_capture_end = time.time()

        # ---- VAC: possibly yields one or more voiced chunks
        yielded = vac.push_frame(frame)
        if not yielded:
            continue

        for voiced_chunk, is_final in yielded:
            if voiced_chunk is None or len(voiced_chunk) == 0:
                continue

            # Update rolling audio buffer only with VOICE
            audio_buf.append(voiced_chunk)
            total_samples_read += len(voiced_chunk)

            # Need minimum audio before decoding
            if audio_buf.n_samples < int(AUDIO_MIN_SEC * sr):
                continue

            window_audio = audio_buf.get_buffer()

            # ---- decode
            t_infer_start = time.time()
            transcribe_kwargs = dict(
                beam_size=int(config.get("beam_size", 5)),
                language=config.get("language", "ar"),
                condition_on_previous_text=False,
                word_timestamps=False,   # not needed anymore (we avoid timestamp barrier)
                no_speech_threshold=float(config.get("no_speech_threshold", 0.6)),
                log_prob_threshold=float(config.get("log_prob_threshold", -1.0)),
                compression_ratio_threshold=float(config.get("compression_ratio_threshold", 2.1)),
                temperature=[0.0, 0.2, 0.4],
                repetition_penalty=float(config.get("repetition_penalty", 1.0)),
            )

            try:
                segments_gen, _info = model.transcribe(window_audio, **transcribe_kwargs)
                segments = list(segments_gen)
            except Exception as e:
                print(f"Whisper Error: {e}")
                continue

            t_infer_end = time.time()
            infer_dur = t_infer_end - t_infer_start
            lag_behind = t_infer_end - t_capture_end

            # Build full hypothesis text
            hyp_text = norm_text("".join([getattr(s, "text", "") for s in segments]))
            if hallucinations_re.search(hyp_text):
                # If the *whole* hypothesis looks polluted, don't advance agreement.
                # (VAC should already reduce this a lot.)
                print(f"[{get_ts()}] REJECTED HALLUCINATION(HYP): {hyp_text}")
                continue

            hyp_words = split_words(hyp_text)

            # Boundary safety: trim last word unless this chunk is FINAL :contentReference[oaicite:12]{index=12}
            if not is_final:
                hyp_words_safe = trim_last_word(hyp_words)
            else:
                hyp_words_safe = hyp_words

            # Also remove hallucination tokens at word level BEFORE agreement
            hyp_words_safe = [w for w in hyp_words_safe if not hallucinations_re.search(w)]

            # ---- LocalAgreement: commit only LCP between previous and current hypotheses :contentReference[oaicite:13]{index=13}
            stable = lcp_words(prev_hyp_words, hyp_words_safe) if prev_hyp_words else []
            prev_hyp_words = hyp_words_safe

            # new committable words = stable suffix beyond what we've already committed
            if len(stable) > len(committed_words):
                new_words = stable[len(committed_words):]

                # small dedupe guard (sometimes first new word repeats last committed)
                if committed_words and new_words and new_words[0] == committed_words[-1]:
                    new_words = new_words[1:]

                if new_words:
                    committed_words.extend(new_words)
                    text_chunk = join_words(new_words)

                    accumulated_text = (accumulated_text + " " + text_chunk).strip()

            # If FINAL: flush everything remaining (even if not in LCP yet)
            if is_final:
                if len(hyp_words_safe) > len(committed_words):
                    tail = hyp_words_safe[len(committed_words):]
                    committed_words = hyp_words_safe[:]  # accept full
                    tail_txt = join_words(tail)
                    if tail_txt:
                        accumulated_text = (accumulated_text + " " + tail_txt).strip()

            # ---- Emit policy
            if accumulated_text and (len(accumulated_text) > 260 or (time.time() - last_send_time > 3.0) or is_final):
                chunk_counter += 1
                clean = accumulated_text.strip()

                log_msg = f"[{get_ts()}] [{chunk_counter}] (Lag: {lag_behind:.2f}s | Infer: {infer_dur:.2f}s) {clean}"
                print(log_msg)
                log_to_file(log_msg)

                trans_q.put({"id": chunk_counter, "text": clean, "ts": time.time()})
                accumulated_text = ""
                last_send_time = time.time()

            # ---- Reset buffers on end-of-voice (this matches SimulStreaming clearing on end-of-voice) :contentReference[oaicite:14]{index=14}
            if is_final:
                audio_buf.clear()
                prev_hyp_words = []
                committed_words = []

    event_q.put(("status", "stream_finished"))
    if wf:
        wf.close()
    if stream:
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
        st.session_state.streaming = False
        st.session_state.extraction_done = False
        st.session_state.stop_event = threading.Event()
        
        st.session_state.refine_in = queue.Queue()
        st.session_state.refine_out = queue.Queue()
        st.session_state.trans_in = queue.Queue()
        st.session_state.event_q = queue.Queue()

    if "refinement_thread_started" not in st.session_state:
        t_ref = threading.Thread(target=refinement_worker, args=(st.session_state.refine_in, st.session_state.refine_out), daemon=True)
        add_script_run_ctx(t_ref)
        t_ref.start()
        st.session_state.refinement_thread_started = True

    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("🗣️ Dialect / Style")
        selected_style = st.selectbox("Select Speech Style", list(PROMPT_TEMPLATES.keys()), index=1)
        st.info(f"Mode: **{selected_style}**")
        st.divider()

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

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.subheader("1. Raw Arabic")
    with c2: st.subheader("2. Refined Arabic")
    with c3: st.subheader("3. Instant English")
    with c4: st.subheader("4. Refined English")
    
    box_raw = c1.empty()
    box_fixed = c2.empty()
    box_draft = c3.empty()
    box_final = c4.empty()

    if source_to_pass is not None:
        if not st.session_state.streaming:
            if st.button("▶️ START STREAM", type="primary", use_container_width=True):
                st.session_state.streaming = True
                st.session_state.stop_event.clear()
                
                config = {
                    "model_size": model_size, 
                    "device": device, 
                    "compute_type": compute_type, 
                    "mt_model": DEFAULT_MT_MODEL, 
                    "refine_every": refine_every, 
                    "prompts": PROMPT_TEMPLATES[selected_style],
                    "logs": {
                        "raw_ar": get_log_path(f"log_{st.session_state.uid}_1_raw_ar.txt"),
                        "draft_en": get_log_path(f"log_{st.session_state.uid}_2_draft_en.txt"),
                        "fixed_ar": get_log_path(f"log_{st.session_state.uid}_3_fixed_ar.txt"),
                        "final_en": get_log_path(f"log_{st.session_state.uid}_4_final_en.txt")
                    }
                }
                
                t1 = threading.Thread(target=instant_translation_worker, args=(st.session_state.trans_in, st.session_state.event_q, config), daemon=True)
                add_script_run_ctx(t1)
                t1.start()

                t2 = threading.Thread(target=transcription_stream_thread, args=(source_to_pass, config, st.session_state.trans_in, st.session_state.stop_event, st.session_state.refine_in, st.session_state.event_q), daemon=True)
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
        box_fixed.text_area("Fixed", value="\n\n".join(st.session_state.refined_blocks_ar), height=600, key="static_fixed")
        box_draft.text_area("Draft", value="\n\n".join([c['tr'] for c in st.session_state.chunks]), height=600, key="static_draft")
        box_final.text_area("Final", value="\n\n".join(st.session_state.refined_blocks_en), height=600, key="static_final")

    if st.session_state.streaming:
        while not st.session_state.stop_event.is_set():
            has_data = False
            try:
                while True:
                    t, p = st.session_state.event_q.get_nowait()
                    if t == "update": st.session_state.chunks.append(p); has_data = True
                    elif t == "error": st.error(f"Error: {p}"); st.session_state.stop_event.set(); break
                    elif t == "status" and p == "stream_finished": st.session_state.stop_event.set(); break
            except queue.Empty: pass

            try:
                while True:
                    p = st.session_state.refine_out.get_nowait()
                    if p["type"] == "refined_batch":
                        st.session_state.refined_blocks_ar.append(f"[{p['id']}] {p['ar_fixed']}")
                        st.session_state.refined_blocks_en.append(f"[{p['id']}] {p['en_final']}")
                        has_data = True
            except queue.Empty: pass
            
            if has_data:
                iter_id = str(uuid.uuid4())[:8] 
                box_raw.text_area("Raw", value="\n\n".join([c['ar'] for c in st.session_state.chunks]), height=600, key=f"raw_{iter_id}")
                box_draft.text_area("Draft", value="\n\n".join([c['tr'] for c in st.session_state.chunks]), height=600, key=f"draft_{iter_id}")
                box_fixed.text_area("Fixed", value="\n\n".join(st.session_state.refined_blocks_ar), height=600, key=f"fixed_{iter_id}")
                box_final.text_area("Final", value="\n\n".join(st.session_state.refined_blocks_en), height=600, key=f"final_{iter_id}")

            time.sleep(0.2) 
        st.rerun()

if __name__ == "__main__":
    main()