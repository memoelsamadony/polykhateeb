"""
Streamlit UI for Khutbah AI real-time transcription.
"""

import os
import time
import uuid
import queue
import tempfile
import threading
import subprocess

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
import sounddevice as sd

from ..config import DEFAULT_MT_MODEL, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from ..utils import log, get_log_path, sanitize_filename
from ..workers import refinement_worker, cloud_polish_worker, transcription_stream_thread
from ..api import telegram_sink_worker
from .shared_state import get_shared_state


def get_input_devices():
    """Get list of available audio input devices."""
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


def extraction_thread(video_path, wav_path, event_q):
    """Extract audio from video file using ffmpeg."""
    try:
        subprocess.run(
            # Cut adhan part: skip first 3 minutes of audio
            ["ffmpeg", "-y", "-ss", "180", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        event_q.put(("status", "extraction_complete"))
    except Exception as e:
        event_q.put(("error", f"FFMPEG Error: {e}"))


def run_app():
    """Main Streamlit application."""
    st.set_page_config(layout="wide", page_title="Khutbah AI")

    # Route to TV viewer mode if requested via ?mode=tv
    if st.query_params.get("mode") == "tv":
        from .tv_viewer import run_tv_viewer
        run_tv_viewer()
        return

    shared = get_shared_state()

    # Initialize session state
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

    # Start background workers
    if "refinement_thread_started" not in st.session_state:
        t_ref = threading.Thread(
            target=refinement_worker,
            args=(st.session_state.refine_in, st.session_state.refine_out, st.session_state.cloud_in),
            daemon=True
        )
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

    # Sidebar configuration
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
                    if name == sel:
                        mic_index = idx
                        break

        st.subheader("🚀 Hardware Optimization")
        model_size = st.selectbox("Whisper Size", ["distil-large-v3", "large-v3", "medium"], index=1)
        compute_type = st.selectbox("Compute Type", ["float16", "int8_float16", "int8"], index=2)
        device = st.radio("Compute Device", ["cuda", "cpu"], index=0)

        refine_every = st.slider("Refine Batch Size", 2, 10, 3)
        if st.button("🔴 RESET APP"):
            st.session_state.stop_event.set()
            shared.reset()
            st.rerun()

    # Main content
    st.title("🕌 Khutbah AI: Real-time Transcription")

    base_logs_path = os.path.join(os.getcwd(), "logs")
    st.caption(f"📂 **LOG FILES SAVING TO:** `{base_logs_path}`")

    source_to_pass = None
    if input_mode == "File Upload":
        u = st.file_uploader("Upload Audio/Video", type=["mp4", "wav"])
        if u and not st.session_state.extraction_done:
            with tempfile.NamedTemporaryFile(delete=False) as t:
                t.write(u.read())
                vp = t.name
            wp = os.path.join(tempfile.gettempdir(), f"audio_{st.session_state.uid}.wav")
            st.session_state.wav_path = wp
            with st.status("Extracting Audio..."):
                extraction_thread(vp, wp, st.session_state.event_q)
                while True:
                    t, m = st.session_state.event_q.get()
                    if t == "status":
                        break
                    if t == "error":
                        st.error(m)
                        st.stop()
                st.session_state.extraction_done = True
            st.rerun()
        if st.session_state.extraction_done:
            source_to_pass = st.session_state.wav_path
    elif mic_index is not None:
        source_to_pass = mic_index

    # Display columns
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.subheader("1. Raw Arabic")
    with c2:
        st.subheader("2. Refined Arabic")
    with c3:
        st.subheader("3. Refined English")
    with c4:
        st.subheader("4. Refined German")
    with c5:
        st.subheader("5. Cloud Translation (EN/DE)")

    box_raw = c1.empty()
    box_fixed = c2.empty()
    box_final_en = c3.empty()
    box_final_de = c4.empty()
    box_cloud = c5.empty()

    # Start/Stop controls
    if source_to_pass is not None:
        if not st.session_state.streaming:
            if st.button("▶️ START STREAM", type="primary", use_container_width=True):
                st.session_state.streaming = True
                st.session_state.stop_event.clear()
                shared.set_streaming(True, uid=st.session_state.uid)

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

                t2 = threading.Thread(
                    target=transcription_stream_thread,
                    args=(
                        source_to_pass,
                        config,
                        st.session_state.stop_event,
                        st.session_state.refine_in,
                        st.session_state.event_q
                    ),
                    daemon=True
                )
                add_script_run_ctx(t2)
                t2.start()
                st.rerun()
        else:
            if st.button("⏹️ STOP STREAM", type="secondary", use_container_width=True):
                st.session_state.stop_event.set()
                st.session_state.streaming = False
                shared.set_streaming(False)
                st.rerun()

    # Static display when not streaming
    if not st.session_state.streaming:
        box_raw.text_area("Raw", value="\n\n".join([c['ar'] for c in st.session_state.chunks]), height=600, key="static_raw")
        box_fixed.text_area("Refined Arabic", value="\n\n".join(st.session_state.refined_blocks_ar), height=600, key="static_fixed")
        box_final_en.text_area("Refined English", value="\n\n".join(st.session_state.refined_blocks_en), height=600, key="static_final_en")
        box_final_de.text_area("Refined German", value="\n\n".join(st.session_state.refined_blocks_de), height=600, key="static_final_de")
        box_cloud.text_area("Cloud Translation", value="\n\n".join(st.session_state.cloud_blocks), height=600, key="static_cloud")

    # Live update loop
    if st.session_state.streaming:
        while not st.session_state.stop_event.is_set():
            has_data = False

            # Process ASR events
            try:
                while True:
                    t, p = st.session_state.event_q.get_nowait()
                    if t == "update":
                        st.session_state.chunks.append(p)
                        st.session_state.last_chunk_id = p.get("id", st.session_state.last_chunk_id)
                        shared.push_chunk(p)
                        has_data = True
                        # Forward to Telegram
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

            # Process refinement results
            try:
                while True:
                    p = st.session_state.refine_out.get_nowait()
                    if p["type"] == "refined_batch":
                        st.session_state.refined_blocks_ar.append(f"[{p['id']}] {p['ar_fixed']}")
                        st.session_state.refined_blocks_en.append(f"[{p['id']}] {p['en_final']}")
                        st.session_state.refined_blocks_de.append(f"[{p['id']}] {p['de_final']}")
                        shared.push_refined(
                            f"[{p['id']}] {p['ar_fixed']}",
                            f"[{p['id']}] {p['en_final']}",
                            f"[{p['id']}] {p['de_final']}",
                        )
                        has_data = True
                        # Forward to Telegram
                        _lat_info = f" (Lat: {p['lat_s']:.2f}s)" if "lat_s" in p else ""
                        st.session_state.telegram_q.put({
                            "text": (
                                f"<b>[Ollama #{p['id']}]{_lat_info}</b>\n"
                                f"<b>AR:</b> {p['ar_fixed']}\n"
                                f"<b>EN:</b> {p['en_final']}\n"
                                f"<b>DE:</b> {p['de_final']}"
                            )
                        })
            except queue.Empty:
                pass

            # Process cloud results
            try:
                while True:
                    p = st.session_state.cloud_out.get_nowait()
                    cloud_text = f"[{p['range'][0]}–{p['range'][1]}] ({p['lang']}, {p['model']})\n{p['text']}"
                    st.session_state.cloud_blocks.append(cloud_text)
                    shared.push_cloud(cloud_text)
                    has_data = True
                    # Forward to Telegram
                    _clat = f" (Lat: {p['lat_s']:.2f}s)" if "lat_s" in p else ""
                    st.session_state.telegram_q.put({
                        "text": (
                            f"<b>[Cloud {p['range'][0]}–{p['range'][1]}] ({p['lang']}, {p['model']}){_clat}</b>\n"
                            f"{p['text']}"
                        )
                    })
            except queue.Empty:
                pass

            # Update UI if data changed
            if has_data:
                iter_id = str(uuid.uuid4())[:8]
                box_raw.text_area("Raw", value="\n\n".join([c['ar'] for c in st.session_state.chunks]), height=600, key=f"raw_{iter_id}")
                box_fixed.text_area("Refined Arabic", value="\n\n".join(st.session_state.refined_blocks_ar), height=600, key=f"fixed_{iter_id}")
                box_final_en.text_area("Refined English", value="\n\n".join(st.session_state.refined_blocks_en), height=600, key=f"final_en_{iter_id}")
                box_final_de.text_area("Refined German", value="\n\n".join(st.session_state.refined_blocks_de), height=600, key=f"final_de_{iter_id}")
                box_cloud.text_area("Cloud Translation", value="\n\n".join(st.session_state.cloud_blocks), height=600, key=f"cloud_{iter_id}")

            time.sleep(0.2)

        # Signal cloud worker on stream end
        if "cloud_in" in st.session_state and st.session_state.cloud_in:
            try:
                st.session_state.cloud_in.put({"id": st.session_state.last_chunk_id, "ar": "", "final": True})
            except Exception:
                pass
        st.rerun()
