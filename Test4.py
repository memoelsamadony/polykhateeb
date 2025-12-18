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
from faster_whisper import WhisperModel
from streamlit.runtime.scriptrunner import add_script_run_ctx

# ==========================================
# 1. CONFIGURATION
# ==========================================

OLLAMA_API_URL = "http://localhost:11434/api/chat"
DEFAULT_MT_MODEL = "qwen2.5:7b-instruct"

# Strict System Prompts
SYSTEM_PROMPT_TRANSLATOR = """ROLE: You are a professional English translator.
TASK: Translate the following Arabic text into English.
RULES:
1. Output ONLY the English translation.
2. Do NOT output Arabic text.
3. Do NOT output Chinese characters.
4. Use "Allah" instead of "God".
5. If the input is incomplete or unclear, output "...".
"""

# NEW PROMPT: Uses Arabic Source for accuracy
SYSTEM_PROMPT_POLISHER = """ROLE: You are an expert Translation Editor.
TASK: Correct and polish the English draft based on the Arabic source.
RULES:
1. Check the Arabic source to fix mistranslations in the draft.
2. Combine sentences into a smooth, natural English paragraph.
3. Fix phonetic errors (e.g. "victims" -> "desires").
4. Output ONLY the polished English text.
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
            output = result.stdout.strip()
            # log(f"[GPU] {tag} | VRAM: {output.replace(chr(10), ' | ')} MB") 
        except: pass

def robust_ollama_call(payload, timeout=30, retries=2):
    for attempt in range(retries + 1):
        try:
            resp = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "").strip()
        except requests.exceptions.Timeout:
            log(f"[API] Timeout (Attempt {attempt+1}). Retrying...")
        except Exception as e:
            log(f"[API] Error: {e}")
        
        if attempt < retries: time.sleep(1)
            
    return None

def validate_translation(original_ar, translated_text):
    if not translated_text or len(translated_text.strip()) < 2:
        return False, "Empty"
    if re.search(r'[\u4e00-\u9fff]', translated_text):
        return False, "Detected Chinese Hallucination"
    ar_chars = len(re.findall(r'[\u0600-\u06FF]', translated_text))
    total_chars = len(translated_text)
    if total_chars > 0 and (ar_chars / total_chars) > 0.40:
        return False, "Echoed Arabic Input"
    return True, "Valid"

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
# 3. RESOURCES
# ==========================================

@st.cache_resource(max_entries=1)
def load_whisper_model(model_size, device, compute_type):
    log_gpu(f"Pre-Load Whisper ({model_size})")
    try:
        gc.collect()
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        log_gpu("Post-Load Whisper")
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
                
                # NOW RECEIVES ARABIC SOURCE TOO
                source_ar = job.get('source_ar', '')
                draft_en = job.get('draft_en', '')
                context_prev = job.get('context', '')
                batch_id = job.get('id', 0)
                log_file = job.get('log_file', None)
                
                if source_ar and draft_en:
                    # NEW PROMPT STRUCTURE
                    prompt = (
                        f"Previous Context (English): {context_prev}\n\n"
                        f"Original Source (Arabic): {source_ar}\n"
                        f"Draft Translation (English): {draft_en}\n\n"
                        f"Task: Correct the English draft based on the Arabic source and rewrite it as a smooth paragraph."
                    )
                    
                    refined = robust_ollama_call({
                        "model": DEFAULT_MT_MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT_POLISHER},
                            {"role": "user", "content": prompt}
                        ],
                        "stream": False,
                        "options": {"temperature": 0.1, "num_ctx": 4096}
                    }, timeout=60, retries=1)
                    
                    if refined:
                        if "Chinese" not in refined and not re.search(r'[\u4e00-\u9fff]', refined):
                            output_q.put(("refined_batch", {"id": batch_id, "text": refined}))
                            log(f"[REFINE] Batch {batch_id} Polished (Used Arabic Source).")
                            if log_file:
                                append_to_file(log_file, f"\n[Batch {batch_id}]\nAR: {source_ar}\nEN: {refined}\n")
                        else:
                            log(f"[REFINE] Batch {batch_id} Rejected (Hallucination).")
                    else:
                        log(f"[REFINE] Batch {batch_id} Failed (Timeout).")
                    
            except queue.Empty:
                continue

    t = threading.Thread(target=run_refinement, daemon=True)
    add_script_run_ctx(t) 
    t.start()
    return input_q, output_q

# ==========================================
# 4. STREAM WORKER
# ==========================================

def extraction_thread(video_path, wav_path, event_q):
    try:
        log("FFMPEG: Extracting audio...")
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
    
    chunk_counter = 0 
    whisper_prompt = "اللهم صل على محمد. خطبة جمعة."
    
    # --- Batching Variables ---
    buffer_ar = [] # Store Arabic chunks
    buffer_tr = [] # Store English chunks
    last_context = ""   
    batch_counter = 0

    trans_log = config['logs']['transcription']
    transl_log = config['logs']['translation']
    refined_log = config['logs']['refined']

    with open(trans_log, "w", encoding="utf-8") as f: f.write("--- ARABIC LOG ---\n")
    with open(transl_log, "w", encoding="utf-8") as f: f.write("--- ENGLISH LOG ---\n")
    with open(refined_log, "w", encoding="utf-8") as f: f.write("--- REFINED LOG ---\n")

    while not stop_event.is_set():
        data = wf.readframes(int(config['chunk_duration'] * sr))
        if not data: break
        
        chunk_counter += 1
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            with wave.open(tmp.name, "wb") as tw:
                tw.setnchannels(wf.getnchannels())
                tw.setsampwidth(wf.getsampwidth())
                tw.setframerate(sr)
                tw.writeframes(data)
            tmp_name = tmp.name
        
        try:
            # === 1. ASR ===
            segments, _ = model.transcribe(
                tmp_name,
                beam_size=config['beam_size'],
                best_of=config['best_of'],
                language="ar",
                initial_prompt=whisper_prompt,
                condition_on_previous_text=False, 
                vad_filter=True,                  
                vad_parameters=dict(min_silence_duration_ms=500),
                temperature=0.0
            )
            
            chunk_ar = " ".join([s.text for s in segments]).strip()
            chunk_ar = re.sub(r'(\b\S+\b\s?)\1{2,}', r'\1', chunk_ar)

            if chunk_ar and len(chunk_ar) > 2:
                whisper_prompt = chunk_ar[-200:]
                log(f"--- [CHUNK {chunk_counter}] ---")
                log(f"AR: {chunk_ar[:40]}...")
                append_to_file(trans_log, f"[{chunk_counter}] {chunk_ar}")

                # === 2. Translation ===
                protected_ar, map_ = protect_terms(chunk_ar)
                
                if "إن الحمد لله" in chunk_ar:
                     raw_tr = "Indeed, all praise is due to Allah..."
                else:
                    raw_tr = robust_ollama_call({
                        "model": config['mt_model'],
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT_TRANSLATOR},
                            {"role": "user", "content": protected_ar}
                        ],
                        "stream": False,
                        "options": {"temperature": 0.1} 
                    }, timeout=40, retries=2)
                    
                    if not raw_tr:
                        raw_tr = "..." 
                    else:
                        is_valid, _ = validate_translation(chunk_ar, raw_tr)
                        if not is_valid: raw_tr = "..." 
                
                final_tr = restore_terms(raw_tr, map_)
                log(f"TR: {final_tr[:40]}...")
                append_to_file(transl_log, f"[{chunk_counter}] {final_tr}")

                event_q.put(("update", {"id": chunk_counter, "ar": chunk_ar, "tr": final_tr}))
                
                # === 3. Batching (Store BOTH AR and TR) ===
                buffer_ar.append(chunk_ar)
                buffer_tr.append(final_tr)
                
                if len(buffer_tr) >= config['refine_every']:
                    batch_counter += 1
                    
                    # Join chunks to form paragraphs
                    batch_ar_text = " ".join(buffer_ar)
                    batch_tr_text = " ".join(buffer_tr)
                    
                    log(f"[BATCH] Sending Batch {batch_counter}")
                    
                    refine_input_q.put({
                        "id": batch_counter,
                        "source_ar": batch_ar_text,  # PASS ARABIC
                        "draft_en": batch_tr_text,   # PASS ENGLISH
                        "context": last_context,
                        "log_file": refined_log
                    })
                    
                    last_context = batch_tr_text[-300:] 
                    buffer_ar = [] 
                    buffer_tr = []

        finally:
            if os.path.exists(tmp_name): os.remove(tmp_name)
            
        if config['simulate']: time.sleep(config['chunk_duration'])

    event_q.put(("status", "stream_finished"))
    wf.close()

# ==========================================
# 5. MAIN UI
# ==========================================

def main():
    st.set_page_config(layout="wide", page_title="Khutbah AI")
    
    refine_in_q, refine_out_q = start_refinement_daemon()
    
    if "uid" not in st.session_state:
        st.session_state.uid = str(uuid.uuid4())[:8]
        st.session_state.chunks = [] 
        st.session_state.refined_blocks = [] 
        st.session_state.event_q = queue.Queue()
        st.session_state.streaming = False
        st.session_state.extraction_done = False
        st.session_state.stop_event = threading.Event()

    with st.sidebar:
        st.header("⚙️ Config")
        model_size = st.selectbox("Whisper Size", ["medium", "large-v2", "large-v3"], index=1)
        device = st.radio("Device", ["cuda", "cpu"], index=0)
        compute_type = st.selectbox("Encoding", ["int8_float16", "float16", "int8"], index=0)

        st.divider()
        beam_size = st.slider("Beam Size", 1, 10, 5)
        best_of = st.slider("Best Of", 1, 10, 5)
        
        st.divider()
        refine_every = st.slider("Refine Batch Size", 1, 5, 3)
        chunk_dur = st.slider("Chunk Duration (s)", 30, 90, 30)
        simulate = st.checkbox("Simulate Realtime", True)
        
        

        if st.button("Reset"):
            st.session_state.stop_event.set()
            st.rerun()

    st.title("🕌 Khutbah AI: Source-Aware Refinement")

    uploaded = st.file_uploader("Upload", type=["mp4", "wav", "mp3"])
    if uploaded and not st.session_state.extraction_done:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded.read())
            video_path = tmp.name
        
        wav_path = os.path.join(tempfile.gettempdir(), f"audio_{st.session_state.uid}.wav")
        st.session_state.wav_path = wav_path
        
        with st.status("Extracting Audio..."):
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
                if st.button("▶️ Start Stream", use_container_width=True):
                    st.session_state.streaming = True
                    st.session_state.stop_event.clear()
                    
                    run_id = st.session_state.uid
                    logs = {
                        "transcription": f"log_{run_id}_transcription.txt",
                        "translation": f"log_{run_id}_translation.txt",
                        "refined": f"log_{run_id}_refined.txt"
                    }
                    st.toast(f"Logs: log_{run_id}_*.txt")

                    config = {
                        "model_size": model_size,
                        "device": device,
                        "compute_type": compute_type,
                        "chunk_duration": chunk_dur,
                        "beam_size": beam_size,
                        "best_of": best_of,
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
                if st.button("⏹️ Stop Stream", use_container_width=True):
                    st.session_state.stop_event.set()
                    st.session_state.streaming = False
                    st.rerun()

        c1, c2, c3 = st.columns(3)
        c1.subheader("Arabic")
        c2.subheader("English (Live)")
        c3.subheader("Refined (Context)")

        while not st.session_state.event_q.empty():
            t, p = st.session_state.event_q.get_nowait()
            if t == "update":
                st.session_state.chunks.append(p)
            elif t == "status": 
                st.session_state.streaming = False
        
        while not refine_out_q.empty():
            t, p = refine_out_q.get_nowait()
            if t == "refined_batch":
                st.session_state.refined_blocks.append(f"[{p['id']}] {p['text']}")

        ar_text = [c['ar'] for c in st.session_state.chunks]
        tr_text = [c['tr'] for c in st.session_state.chunks]
            
        with c1: st.text_area("AR", value="\n\n".join(ar_text), height=600)
        with c2: st.text_area("TR", value="\n\n".join(tr_text), height=600)
        with c3: st.text_area("Refined", value="\n\n".join(st.session_state.refined_blocks), height=600)

        if st.session_state.streaming:
            time.sleep(1)
            st.rerun()

if __name__ == "__main__":
    main()