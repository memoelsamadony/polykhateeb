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
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from streamlit.runtime.scriptrunner import add_script_run_ctx

# ==========================================
# 1. CONFIGURATION & PROMPTS
# ==========================================

OLLAMA_API_URL = "http://localhost:11434/api/chat"
# Default to the 3B model for GPU fit
DEFAULT_MT_MODEL = "qwen2.5:3b-instruct"

PROMPT_TEMPLATES = {
    "Standard (Fusha/MSA)": {
        "fixer": """TASK: Correct Arabic typos.
INPUT: Formal Khutbah.
RULES:
1. Output ONLY the corrected Arabic text.
2. NO "Here is the corrected text". NO intro. NO notes.
3. Fix grammar/spelling only.
4. If modern terms appear, preserve them.""",
        "translator": """TASK: Translate to English.
STYLE: Formal, Biblical/Sermon-like.
RULES:
1. Output ONLY the translation.
2. NO conversational filler.
3. Start directly with the translated text."""
    },
    
    "Egyptian (Masri)": {
        "fixer": """TASK: Fix typos.
CONTEXT: Egyptian Dialect with potential English loanwords.
RULES:
1. RESPECT DIALECT: Keep words like "Basita", "Keda", "Yansoon" EXACTLY as is.
2. LOANWORDS: Expect English words written in Arabic (e.g. "كاميكلز" -> Chemicals, "كورس" -> Course). Do NOT change them to random Arabic words.
3. Output ONLY the corrected Arabic.
4. NO "Corrected text:" prefixes.""",
        "translator": """TASK: Translate to English.
STYLE: Casual/Storytelling.
RULES:
1. Output ONLY the translation.
2. NO "Here is...", NO "Sure...", NO "Note:".
3. DETECT LOANWORDS: If you see Arabized English (e.g., "Kamiklz"), translate it to the English equivalent ("Chemicals")."""
    },
    
    "Gulf (Khaleeji)": {
        "fixer": """TASK: Fix typos.
CONTEXT: Gulf Dialect with potential English loanwords.
RULES:
1. Keep Khaleeji words (Shlonak, Zein) EXACTLY as is.
2. LOANWORDS: Handle English terms (e.g. "Absher", "System") correctly.
3. Output ONLY the corrected Arabic.
4. NO conversational filler.""",
        "translator": """TASK: Translate to English.
STYLE: Respectful/Natural.
RULES:
1. Output ONLY the translation.
2. NO preamble. NO notes.
3. Handle English loanwords naturally."""
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
    """Creates a 'logs' folder in the current script directory."""
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

def robust_ollama_call(payload, timeout=30, retries=1):
    """Sends request to Ollama with cleaning to strip 'Sure!'/preambles."""
    for attempt in range(retries + 1):
        try:
            resp = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            text = resp.json().get("message", {}).get("content", "").strip()
            
            # --- CLEANER ---
            # Remove <think> tags (deepseek support)
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            # Remove chatty prefixes
            text = re.sub(r'^(Here is|Sure|Translation|Corrected|Okay|I have translated).*?:', '', text, flags=re.IGNORECASE).strip()
            # Remove notes at the end
            text = re.sub(r'(Note:|P.S.|Analysis:).*', '', text, flags=re.IGNORECASE).strip()
            # Remove surrounding quotes
            text = text.strip('"').strip("'")
            # ---------------

            if re.search(r'[\u4e00-\u9fff]', text): return None # Reject Chinese hallucinations
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
    
    current_fixer_prompt = PROMPT_TEMPLATES["Standard (Fusha/MSA)"]["fixer"]
    current_final_prompt = PROMPT_TEMPLATES["Standard (Fusha/MSA)"]["translator"]

    while True:
        try:
            job = input_q.get(timeout=2) 
            if job is None: continue 
            if job == "STOP": break

            raw_ar = job.get('source_ar', '')
            batch_id = job.get('id', 0)
            log_file_ar = job.get('log_file_ar', None)
            log_file_en = job.get('log_file_en', None)
            t_submitted = job.get('ts', time.time())
            
            if 'prompts' in job:
                current_fixer_prompt = job['prompts']['fixer']
                current_final_prompt = job['prompts']['translator']

            log(f"Refining Batch {batch_id}...")

            if raw_ar:
                # 1. Fix Arabic
                corrected_ar = robust_ollama_call({
                    "model": DEFAULT_MT_MODEL,
                    "messages": [{"role": "system", "content": current_fixer_prompt},{"role": "user", "content": raw_ar}],
                    "stream": False, "options": {"temperature": 0.1, "num_ctx": 1024}
                }, timeout=50)

                if not corrected_ar: corrected_ar = raw_ar 
                
                t_fixed = time.time()
                lat_fix = t_fixed - t_submitted
                if log_file_ar: 
                    append_to_file(log_file_ar, f"[{get_current_ts_string()}] [Batch {batch_id}] (Lat: {lat_fix:.2f}s)\n{corrected_ar}\n")

                # 2. Final Translate
                final_en = robust_ollama_call({
                    "model": DEFAULT_MT_MODEL,
                    "messages": [{"role": "system", "content": current_final_prompt},{"role": "user", "content": f"Corrected Arabic: {corrected_ar}"}],
                    "stream": False, "options": {"temperature": 0.3, "num_ctx": 1024}
                }, timeout=50)

                if final_en:
                    t_final = time.time()
                    lat_final = t_final - t_submitted
                    
                    output_q.put({
                        "type": "refined_batch",
                        "id": batch_id, 
                        "ar_fixed": corrected_ar,
                        "en_final": final_en
                    })
                    if log_file_en: 
                        append_to_file(log_file_en, f"[{get_current_ts_string()}] [Batch {batch_id}] (Total Lat: {lat_final:.2f}s)\n{final_en}\n")
                    
                    log(f"Batch {batch_id} Done (Lat: {lat_final:.2f}s).")
        except queue.Empty: continue
        except Exception as e:
            log(f"Refinement Worker Crash: {e}"); time.sleep(1)

# --- WORKER 2: INSTANT TRANSLATION ---
def instant_translation_worker(transcription_q, event_q, config):
    log("Instant Translator: STARTING")
    translator_prompt = config['prompts']['translator']
    
    while True:
        try:
            item = transcription_q.get(timeout=1)
            if item is None: break 
            
            chunk_id = item['id']
            ar_text = item['text']
            t_emit = item.get('ts', time.time())
            log_draft_en = config['logs']['draft_en']

            protected, map_ = protect_terms(ar_text)
            raw_tr = robust_ollama_call({
                "model": config['mt_model'],
                "messages": [{"role": "system", "content": translator_prompt},{"role": "user", "content": protected}],
                "stream": False, "options": {"temperature": 0.1, "num_ctx": 1024}
            }, timeout=45) or "..."
            
            final_tr = restore_terms(raw_tr, map_)
            
            t_now = time.time()
            latency = t_now - t_emit
            
            append_to_file(log_draft_en, f"[{get_current_ts_string()}] [{chunk_id}] (Lat: {latency:.2f}s) {final_tr}")

            event_q.put(("update", {"id": chunk_id, "ar": ar_text, "tr": final_tr}))
        except queue.Empty: continue
        except Exception as e: log(f"Translator Error: {e}")

# --- WORKER 3: AUDIO & WHISPER (Tuned) ---
# --- WORKER 3: AUDIO & WHISPER (VAD DISABLED + HALLUCINATION FIX) ---
def transcription_stream_thread(source, config, trans_q, stop_event, refine_input_q, event_q):
    # 1. Load Model
    model = load_whisper_model(config['model_size'], config['device'], config['compute_type'])
    if not model: event_q.put(("error", "Model failed to load.")); return

    # 2. Setup Stream
    is_live_mic = isinstance(source, int)
    wf, stream = None, None
    sr = 16000
    
    if is_live_mic:
        try:
            # Standard blocksize for responsiveness
            stream = sd.InputStream(device=source, channels=1, samplerate=sr, dtype="float32", blocksize=int(sr * 0.5))
            stream.start()
        except Exception as e: event_q.put(("error", f"Mic Error: {e}")); return
    else:
        try: wf = wave.open(source, "rb"); sr = wf.getframerate()
        except Exception as e: event_q.put(("error", f"File Error: {e}")); return
    
    audio_buffer = np.array([], dtype=np.float32)
    chunk_counter = 0
    accumulated_ar_text = ""
    last_send_time = time.time()
    
    # SETTING: Wait for full phrases
    MIN_SEND_LENGTH = 100 
    MAX_WAIT_TIME = 8.0
    
    # SETTING: Strong Religious Priming to prevent "Nancy"
    whisper_prompt = "خطبة جمعة. الله. الرسول. القرآن. قال الله تعالى. بسم الله الرحمن الرحيم."
    
    refine_buffer_ar = []
    batch_counter = 0
    prompts_config = config['prompts']
    
    # Logs setup
    log_raw_ar = config['logs']['raw_ar']
    log_fixed_ar = config['logs']['fixed_ar']
    log_final_en = config['logs']['final_en']
    log_draft_en = config['logs']['draft_en']

    for p in [log_raw_ar, log_draft_en, log_fixed_ar, log_final_en]:
        try:
            with open(p, "w", encoding="utf-8") as f: 
                f.write(f"--- LOG START: {datetime.datetime.now()} ---\nFORMAT: [TIMESTAMP] [ID] (Latency Info) Text\n\n")
        except: pass

    READ_CHUNK_SEC = 1.0 if is_live_mic else 0.5
    
    while not stop_event.is_set():
        new_audio = np.array([], dtype=np.float32)
        if is_live_mic:
            if stream.active:
                data, overflow = stream.read(int(READ_CHUNK_SEC * sr))
                new_audio = data.flatten()
            else: break
        else:
            raw_bytes = wf.readframes(int(READ_CHUNK_SEC * sr))
            if not raw_bytes: break
            new_audio = np.frombuffer(raw_bytes, np.int16).flatten().astype(np.float32) / 32768.0

        t_capture_finished = time.time()
        audio_buffer = np.concatenate((audio_buffer, new_audio))
        
        # Max buffer 30s
        if len(audio_buffer) > 30 * sr: audio_buffer = audio_buffer[-30*sr:] 
        
        # --- FIX 1: INCREASE BUFFER SIZE ---
        # Wait for 3.0 seconds of audio. 
        # Mosques have pauses; 1s is too short and triggers "Silence Hallucinations".
        if len(audio_buffer) < 3.0 * sr: continue

        try:
            t_infer_start = time.time()
            
            segments, _ = model.transcribe(
                audio_buffer, 
                beam_size=5,
                best_of=5,
                language="ar", 
                task="transcribe", 
                initial_prompt=whisper_prompt,
                condition_on_previous_text=False,
                
                # --- FIX 2: DISABLE VAD ---
                # We turn this OFF because mosque echo confuses it.
                vad_filter=False, 
                
                word_timestamps=True,
                repetition_penalty=1.2,
                # Higher threshold means "Don't output text unless you are SURE it is speech"
                no_speech_threshold=0.6 
            )
            
            t_infer_end = time.time()
            inference_duration = t_infer_end - t_infer_start
            lag_behind_speaker = t_infer_end - t_capture_finished

            current_words = [w for s in segments for w in s.words]
            if not current_words: 
                # If model says nothing, cut a small chunk and move on
                cut_sample = int(1.0 * sr)
                audio_buffer = audio_buffer[cut_sample:]
                continue

            UNSTABLE_REGION_SEC = 1.0 
            cutoff_time = (len(audio_buffer) / sr) - UNSTABLE_REGION_SEC
            
            committed_words = []
            final_timestamp_end = 0.0
            
            for w in current_words:
                if w.end <= cutoff_time:
                    committed_words.append(w.word)
                    final_timestamp_end = w.end
                else: break
            
            if committed_words:
                new_text_segment = "".join(committed_words).strip()
                
                # --- FIX 3: THE "NANCY" KILLER ---
                # Regex to nuke common subtitle hallucinations
                hallucinations = r'(Subtitle|Translated|Amara|MBC|Nancy|Nana|Copyright|Rights|Reserved|Music|Unidentified)'
                if re.search(hallucinations, new_text_segment, re.IGNORECASE):
                    log(f"HALLUCINATION REMOVED: {new_text_segment}")
                    # If the WHOLE chunk is a hallucination, dump the audio buffer
                    cut_sample = int(final_timestamp_end * sr)
                    audio_buffer = audio_buffer[cut_sample:]
                    continue
                
                # Filter meaningless short garbage (e.g. ".")
                if len(new_text_segment) < 2: 
                    cut_sample = int(final_timestamp_end * sr)
                    audio_buffer = audio_buffer[cut_sample:]
                    continue

                accumulated_ar_text += " " + new_text_segment
                time_since_send = time.time() - last_send_time
                is_long_enough = len(accumulated_ar_text) > MIN_SEND_LENGTH
                is_timeout = time_since_send > MAX_WAIT_TIME
                
                if is_long_enough or (is_timeout and len(accumulated_ar_text) > 5):
                    chunk_counter += 1
                    clean_text = accumulated_ar_text.strip()
                    t_emit = time.time()
                    
                    log_msg = f"SENDING: {clean_text[:30]}... (Infer: {inference_duration:.2f}s | Lag: {lag_behind_speaker:.2f}s)"
                    log(log_msg)
                    
                    append_to_file(log_raw_ar, f"[{get_current_ts_string()}] [{chunk_counter}] (Lag: {lag_behind_speaker:.2f}s | Infer: {inference_duration:.2f}s) {clean_text}")

                    trans_q.put({"id": chunk_counter, "text": clean_text, "ts": t_emit})
                    
                    # Refinement Batching
                    refine_buffer_ar.append(clean_text)
                    if len(refine_buffer_ar) >= config['refine_every']:
                        batch_counter += 1
                        batch_ar = " ".join(refine_buffer_ar)
                        refine_input_q.put({
                            "id": batch_counter,
                            "source_ar": batch_ar,
                            "prompts": prompts_config,
                            "ts": time.time(),
                            "log_file_ar": log_fixed_ar,
                            "log_file_en": log_final_en 
                        })
                        refine_buffer_ar = []

                    accumulated_ar_text = ""
                    last_send_time = time.time()
                    whisper_prompt = clean_text[-200:]

                cut_sample = int(final_timestamp_end * sr)
                audio_buffer = audio_buffer[cut_sample:]
            else:
                # If no words were committed but inference ran, check if buffer is getting too full
                if len(audio_buffer) > 10 * sr:
                     audio_buffer = audio_buffer[-5*sr:]

        except Exception as e: log(f"Whisper Error: {e}")
            
    event_q.put(("status", "stream_finished"))
    if wf: wf.close()
    if stream: stream.close()

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
        
        # --- VRAM OPTIMIZATION SETTINGS ---
        st.subheader("🚀 Hardware Optimization")
        model_size = st.selectbox("Whisper Size", ["distil-large-v3", "large-v3", "medium"], index=1, help="Use 'large-v3' for best Arabic.")
        compute_type = st.selectbox("Compute Type", ["float16", "int8_float16", "int8"], index=2, help="Use 'int8' for 6GB GPU.")
        device = st.radio("Compute Device", ["cuda", "cpu"], index=0)
        
        refine_every = st.slider("Refine Batch Size", 2, 10, 3)
        if st.button("🔴 RESET APP"): st.session_state.stop_event.set(); st.rerun()

    st.title("🕌 Khutbah AI: Real-time Transcription")
    
    # Path Display
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
                    "compute_type": compute_type, # Use selected int8
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