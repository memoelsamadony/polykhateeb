import os
import subprocess
import tempfile
import wave
import numpy as np
import streamlit as st
from faster_whisper import WhisperModel

# ---------------------------------------------
# Helpers
# ---------------------------------------------

@st.cache_resource(max_entries=1)
def load_whisper_model(model_size: str, device: str, compute_type: str):
    id_map = {
        "distil-large-v3": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "large-v3": "Systran/faster-whisper-large-v3",
        "medium": "Systran/faster-whisper-medium",
    }
    hf_id = id_map.get(model_size, model_size)
    return WhisperModel(hf_id, device=device, compute_type=compute_type)


def extract_audio_to_wav(input_path: str, target_path: str, sample_rate: int = 16000):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        target_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def load_first_half_audio(wav_path: str) -> tuple[np.ndarray, int]:
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        half_frames = n_frames // 2
        wf.rewind()
        raw = wf.readframes(half_frames)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def transcribe_first_half(audio: np.ndarray, model: WhisperModel, beam_size: int):
    segments, info = model.transcribe(
        audio,
        language="ar",
        task="transcribe",
        beam_size=beam_size,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.1,
        temperature=[0.0, 0.2, 0.4],
    )
    text_parts = [s.text for s in segments]
    return "".join(text_parts).strip(), info


# ---------------------------------------------
# UI
# ---------------------------------------------

st.set_page_config(layout="wide", page_title="Half Audio Transcriber")
st.title("🕌 Half-File Transcription (Arabic half → Whisper)")
st.caption("Uploads the full file, splits audio in half, and only sends the first half to Whisper (no VAD, no initial prompt).")

with st.sidebar:
    st.header("⚙️ Whisper Settings")
    model_size = st.selectbox("Whisper Size", ["distil-large-v3", "large-v3", "medium"], index=1)
    compute_type = st.selectbox("Compute Type", ["float16", "int8_float16", "int8"], index=2)
    device = st.radio("Device", ["cuda", "cpu"], index=0)
    beam_size = st.slider("Beam Size", 1, 8, 5)

uploader = st.file_uploader("Upload audio/video (first half Arabic, second half German)", type=["mp4", "wav", "mp3", "m4a", "aac"])

if uploader:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploader.name)[1]) as tmp_in:
            tmp_in.write(uploader.read())
            input_path = tmp_in.name

        wav_path = os.path.join(tempfile.gettempdir(), f"half_audio_{os.path.basename(input_path)}.wav")

        with st.spinner("Extracting audio..."):
            extract_audio_to_wav(input_path, wav_path)

        with st.spinner("Loading Whisper model..."):
            model = load_whisper_model(model_size, device, compute_type)

        with st.spinner("Reading first half of audio..."):
            audio, sr = load_first_half_audio(wav_path)

        with st.spinner("Transcribing first half (Arabic)..."):
            text, info = transcribe_first_half(audio, model=model, beam_size=beam_size)

        st.success("Done. Only the first half was transcribed.")
        st.write(f"Sample rate: {sr} Hz | First-half duration: {len(audio) / sr:.2f} s")
        st.text_area("Transcript (Arabic first half)", value=text, height=320)

    except subprocess.CalledProcessError:
        st.error("FFmpeg failed to extract audio. Ensure FFmpeg is installed and the file is valid.")
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass
        try:
            os.remove(input_path)
        except Exception:
            pass
else:
    st.info("Upload a file to begin. The app will split the audio in half and only send the first half to Whisper.")
