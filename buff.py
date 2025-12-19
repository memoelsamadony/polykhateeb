import streamlit as st
import threading
import queue
import time
import os
import subprocess
import wave
import tempfile
import re
import requests
import shutil
import uuid
import gc
import numpy as np
from faster_whisper import WhisperModel
from streamlit.runtime.scriptrunner import add_script_run_ctx

# ==========================================
# 1. CONFIGURATION
# ==========================================

OLLAMA_API_URL = "http://localhost:11434/api/chat"
DEFAULT_MT_MODEL = "qwen2.5:7b-instruct"

# Global Lock
GPU_LOCK = threading.Lock()

# --- PROMPTS ---
SYSTEM_PROMPT_TRANSLATOR = """ROLE: Translator.
TASK: Translate Arabic to English.
RULES:
1. Output ONLY English.
2. Use "Allah" for God.
3. If unclear, output "...".
"""

SYSTEM_PROMPT_AR_FIXER = """ROLE: Arabic Editor.
TASK: Correct ASR errors.
RULES:
1. Fix typos (e.g. 'أرمين' -> 'آمين').
2. Remove non-Arabic text/hallucinations.
3. Return ONLY corrected Arabic.
"""

SYSTEM_PROMPT_FINAL = """ROLE: Islamic Translator.
TASK: Translate Corrected Arabic to English.
RULES:
1. Ground truth: Corrected Arabic.
2. Tone: Sermon-like.
3. Fix phonetic errors.
4. Output ONLY English.
"""

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

def log(msg):
    print(f"[{time.strftime('%X')}] {msg}", flush=True)

def append_to_file(filepath, text):
    if not text: return
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception as e:
        log(f"FILE ERROR: {e}")

def log_gpu(tag: str):
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
                capture_output=True, text=True
            )
            # log(f"[GPU] {tag} | VRAM: {result.stdout.strip()}") 
        except: pass

def robust_ollama_call(payload, timeout=30, retries=1):
    for attempt in range(retries + 1):
        try:
            with GPU_LOCK:
                resp = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            text = resp.json().get("message", {}).get("content", "").strip()
            if re.search(r'[\u4e00-\u9fff]', text): return None
            return text
        except requests.exceptions.Timeout:
            log(f"[API] Timeout (Attempt {attempt+1})...")
        except Exception as e:
            log(f"[API] Error: {e}")
        if attempt < retries: time.sleep(1)
    return None

def validate_translation(text):
    if not text or len(text.strip()) < 2: return False
    if re.search(r'[\u4e00-\u9fff]', text): return False
    return True

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
# 3. RESOURCES & DAEMON
# ==========================================

@st.cache_resource(max_entries=1)
def load_whisper_model(model_size, device, compute_type):
    if model_size == "distil-large-v3": hf_id = "distil-whisper/distil-large-v3"
    else: hf_id = model_size
    log(f"Loading Whisper: {hf_id}...")
    try:
        gc.collect()
        model = WhisperModel(hf_id, device=device, compute_type=compute_type)
        log("Whisper Loaded.")
        return model
    except Exception as e:
        log(f"CRITICAL: Whisper Load Failed: {e}")
        return None

@st.cache_resource
def start_refinement_daemon():
    input_q = queue.Queue()
    output_q = queue.Queue()
    
    def run_refinement():
        log("Refinement Daemon: ONLINE")
        while True:
            try:
                job = input_q.get(timeout=1)
                if job is None: break
                
                raw_ar = job.get('source_ar', '')
                batch_id = job.get('id', 0)
                log_file_ar = job.get('log_file_ar', None)
                log_file_en = job.get('log_file_en', None)
                
                if raw_ar:
                    # 1. Correct Arabic
                    corrected_ar = robust_ollama_call({
                        "model": DEFAULT_MT_MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT_AR_FIXER},
                            {"role": "user", "content": raw_ar}
                        ],
                        "stream": False,
                        "options": {"temperature": 0.1, "num_ctx": 2048}
                    }, timeout=50)

                    if not corrected_ar: corrected_ar = raw_ar 
                    if log_file_ar: append_to_file(log_file_ar, f"\n[Batch {batch_id}]\nFIXED: {corrected_ar}\n")

                    # 2. Translate Corrected
                    final_en = robust_ollama_call({
                        "model": DEFAULT_MT_MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT_FINAL},
                            {"role": "user", "content": f"Corrected Arabic: {corrected_ar}"}
                        ],
                        "stream": False,
                        "options": {"temperature": 0.3, "num_ctx": 2048}
                    }, timeout=50)

                    if final_en and validate_translation(final_en):
                        output_q.put({
                            "type": "refined_batch",
                            "id": batch_id, 
                            "ar_fixed": corrected_ar,
                            "en_final": final_en
                        })
                        if log_file_en: append_to_file(log_file_en, f"\n[Batch {batch_id}]\n{final_en}\n")
                        log(f"[BATCH {batch_id}] DONE.")
                    else:
                        log(f"[BATCH {batch_id}] FAILED.")
            except queue.Empty: continue

    t = threading.Thread(target=run_refinement, daemon=True)
    add_script_run_ctx(t) 
    t.start()
    return input_q, output_q

# ==========================================
# 4. STREAM WORKER (LocalAgreement Policy)
# ==========================================

def extraction_thread(video_path, wav_path, event_q):
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            wav_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        event_q.put(("status", "extraction_complete"))
    except Exception as e:
        event_q.put(("error", f"FFMPEG Error: {e}"))

def transcription_stream_thread(wav_path, config, event_q, stop_event, refine_input_q):
    model = load_whisper_model(config['model_size'], config['device'], config['compute_type'])
    if not model:
        event_q.put(("error", "Model failed to load."))
        return

    wf = wave.open(wav_path, "rb")
    sr = wf.getframerate()
    
    # --- LocalAgreement State ---
    audio_buffer = np.array([], dtype=np.float32) # Rolling buffer
    last_committed_text = ""
    chunk_counter = 0
    whisper_prompt = "اللهم صل على محمد. خطبة جمعة."
    
    # Refinement Buffers
    refine_buffer_ar = []
    refine_buffer_tr = []
    batch_counter = 0

    log_raw_ar = config['logs']['raw_ar']
    log_draft_en = config['logs']['draft_en']
    log_fixed_ar = config['logs']['fixed_ar']
    log_final_en = config['logs']['final_en']

    for p in [log_raw_ar, log_draft_en, log_fixed_ar, log_final_en]:
        with open(p, "w", encoding="utf-8") as f: f.write(f"--- LOG START ---\n")

    # Read smaller chunks frequently (e.g., 2s) to update buffer
    STEP_SIZE_SEC = 3.0 
    
    while not stop_event.is_set():
        # Read new audio bytes
        raw_bytes = wf.readframes(int(STEP_SIZE_SEC * sr))
        if not raw_bytes: break
        
        # Convert to Float32 for Whisper
        new_audio = np.frombuffer(raw_bytes, np.int16).flatten().astype(np.float32) / 32768.0
        audio_buffer = np.concatenate((audio_buffer, new_audio))
        
        # Max buffer safety (prevent infinite growth)
        if len(audio_buffer) > 30 * sr:
            audio_buffer = audio_buffer[-30*sr:] # Keep last 30s max

        try:
            with GPU_LOCK:
                # Transcribe the WHOLE buffer
                segments, _ = model.transcribe(
                    audio_buffer,
                    beam_size=1, best_of=1,
                    language="ar", initial_prompt=whisper_prompt,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500, threshold=0.6),
                    word_timestamps=True # CRITICAL for slicing
                )
            
            # Extract words and timestamps
            current_words = []
            for s in segments:
                for w in s.words:
                    current_words.append(w)
            
            if not current_words: continue

            # --- POLICY: "Stable Hold-Back" ---
            # We commit everything EXCEPT the last X seconds (unstable region)
            # Unless buffer is getting too full, then we force commit.
            
            UNSTABLE_REGION_SEC = 1.5 
            buffer_duration = len(audio_buffer) / sr
            
            # Determine cutoff timestamp
            cutoff_time = buffer_duration - UNSTABLE_REGION_SEC
            
            committed_words = []
            final_timestamp_end = 0.0
            
            for w in current_words:
                if w.end <= cutoff_time:
                    committed_words.append(w.word)
                    final_timestamp_end = w.end
                else:
                    break # Stop at unstable region
            
            # If we have substantial text to commit
            if committed_words:
                chunk_counter += 1
                new_text_segment = "".join(committed_words).strip()
                
                # Update Whisper Prompt
                whisper_prompt = new_text_segment[-200:]
                
                log(f"--- [CHUNK {chunk_counter}] ---")
                log(f"AR: {new_text_segment[:40]}...")
                append_to_file(log_raw_ar, f"[{chunk_counter}] {new_text_segment}")

                # Translate Committed Chunk
                protected, map_ = protect_terms(new_text_segment)
                raw_tr = robust_ollama_call({
                    "model": config['mt_model'],
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT_TRANSLATOR},
                        {"role": "user", "content": protected}
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1}
                }, timeout=20) or "..."
                
                final_tr = restore_terms(raw_tr, map_)
                append_to_file(log_draft_en, f"[{chunk_counter}] {final_tr}")
                
                event_q.put(("update", {"id": chunk_counter, "ar": new_text_segment, "tr": final_tr}))

                # Remove committed audio from buffer
                # Sample index = seconds * sample_rate
                cut_sample = int(final_timestamp_end * sr)
                audio_buffer = audio_buffer[cut_sample:]
                
                # Add to Batch Buffer
                refine_buffer_ar.append(new_text_segment)
                
                if len(refine_buffer_ar) >= config['refine_every']:
                    batch_counter += 1
                    batch_ar = " ".join(refine_buffer_ar)
                    refine_input_q.put({
                        "id": batch_counter,
                        "source_ar": batch_ar,
                        "log_file_ar": log_fixed_ar,
                        "log_file_en": log_final_en 
                    })
                    refine_buffer_ar = []

        except Exception as e:
            log(f"Stream Loop Error: {e}")
            
    event_q.put(("status", "stream_finished"))
    wf.close()

# ==========================================
# 5. MAIN UI
# ==========================================

def main():
    st.set_page_config(layout="wide", page_title="Khutbah AI (Simul)")
    
    refine_in_q, refine_out_q = start_refinement_daemon()
    
    if "uid" not in st.session_state:
        st.session_state.uid = str(uuid.uuid4())[:8]
        st.session_state.chunks = [] 
        st.session_state.refined_blocks_ar = [] 
        st.session_state.refined_blocks_en = [] 
        st.session_state.event_q = queue.Queue()
        st.session_state.streaming = False
        st.session_state.extraction_done = False
        st.session_state.stop_event = threading.Event()

    with st.sidebar:
        st.header("⚙️ Settings")
        model_size = st.selectbox("Whisper", ["distil-large-v3", "large-v3", "medium"], index=0)
        device = st.radio("Device", ["cuda", "cpu"], index=0)
        compute_type = st.selectbox("Encoding", ["int8_float16", "float16", "int8"], index=0)
        refine_every = st.slider("Refine Batch", 2, 6, 3)
        chunk_dur = st.slider("Update Rate (s)", 2, 10, 3, help="How often to process buffer")
        simulate = st.checkbox("Realtime Sim", True)
        
        if st.button("Reset"):
            st.session_state.stop_event.set()
            st.rerun()

    st.title("🕌 Khutbah AI: SimulStreaming Policy")

    uploaded = st.file_uploader("Upload", type=["mp4", "wav", "mp3"])
    if uploaded and not st.session_state.extraction_done:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded.read())
            video_path = tmp.name
        
        wav_path = os.path.join(tempfile.gettempdir(), f"audio_{st.session_state.uid}.wav")
        st.session_state.wav_path = wav_path
        
        with st.status("Processing..."):
            extraction_thread(video_path, wav_path, st.session_state.event_q)
            while True:
                t, m = st.session_state.event_q.get()
                if t == "status": break
                if t == "error": st.error(m); st.stop()
            st.session_state.extraction_done = True
        st.rerun()

    if st.session_state.extraction_done:
        col1, col2 = st.columns([1, 4])
        with col1:
            if not st.session_state.streaming:
                if st.button("▶️ Start", use_container_width=True):
                    st.session_state.streaming = True
                    st.session_state.stop_event.clear()
                    
                    run_id = st.session_state.uid
                    logs = {
                        "raw_ar": f"log_{run_id}_1_RAW_AR.txt",
                        "draft_en": f"log_{run_id}_2_DRAFT_EN.txt",
                        "fixed_ar": f"log_{run_id}_3_FIXED_AR.txt",
                        "final_en": f"log_{run_id}_4_FINAL_EN.txt"
                    }
                    st.toast(f"Logs: log_{run_id}_*.txt")

                    config = {
                        "model_size": model_size,
                        "device": device,
                        "compute_type": compute_type,
                        "chunk_duration": chunk_dur,
                        "beam_size": 1, 
                        "best_of": 1,
                        "mt_model": DEFAULT_MT_MODEL,
                        "refine_every": refine_every,
                        "simulate": simulate,
                        "logs": logs 
                    }
                    
                    t = threading.Thread(
                        target=transcription_stream_thread,
                        args=(st.session_state.wav_path, config, st.session_state.event_q, st.session_state.stop_event, refine_in_q),
                        daemon=True
                    )
                    add_script_run_ctx(t) 
                    t.start()
                    st.rerun()
            else:
                if st.button("⏹️ Stop", use_container_width=True):
                    st.session_state.stop_event.set()
                    st.session_state.streaming = False
                    st.rerun()

        c1, c2, c3, c4 = st.columns(4)
        c1.subheader("1. Raw Arabic")
        c2.subheader("2. Corrected Arabic")
        c3.subheader("3. Draft English")
        c4.subheader("4. Final English")

        while not st.session_state.event_q.empty():
            t, p = st.session_state.event_q.get_nowait()
            if t == "update":
                st.session_state.chunks.append(p)
            elif t == "status": 
                st.session_state.streaming = False
        
        while not refine_out_q.empty():
            payload = refine_out_q.get_nowait()
            if payload.get("type") == "refined_batch":
                st.session_state.refined_blocks_ar.append(f"[{payload['id']}] {payload['ar_fixed']}")
                st.session_state.refined_blocks_en.append(f"[{payload['id']}] {payload['en_final']}")

        raw_ar_text = [c['ar'] for c in st.session_state.chunks]
        draft_en_text = [c['tr'] for c in st.session_state.chunks]
            
        with c1: st.text_area("Raw", value="\n\n".join(raw_ar_text), height=600)
        with c2: st.text_area("Fixed", value="\n\n".join(st.session_state.refined_blocks_ar), height=600)
        with c3: st.text_area("Draft", value="\n\n".join(draft_en_text), height=600)
        with c4: st.text_area("Final", value="\n\n".join(st.session_state.refined_blocks_en), height=600)

        if st.session_state.streaming:
            time.sleep(1)
            st.rerun()

if __name__ == "__main__":
    main()