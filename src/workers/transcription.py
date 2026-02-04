"""
Transcription worker using NVIDIA NeMo FastConformer.

Handles real-time Arabic speech recognition with VAD-based segmentation.
"""

import time
import datetime
import queue
import tempfile
import threading
import numpy as np

from ..utils import append_to_file


def transcription_stream_thread(
    source,
    config: dict,
    stop_event: threading.Event,
    refine_input_q: queue.Queue,
    event_q: queue.Queue,
) -> None:
    """
    NVIDIA FastConformer (RNNT/Transducer Mode) transcription worker.

    Handles both live microphone and file-based audio sources.
    Uses VAD for intelligent segmentation.

    Args:
        source: Microphone device index (int) or WAV file path (str)
        config: Configuration dict with model settings and log paths
        stop_event: Threading event to signal stop
        refine_input_q: Queue to send transcription results for refinement
        event_q: Queue to send UI events
    """
    import wave
    import torch
    import nemo.collections.asr as nemo_asr
    import soundfile as sf
    import sounddevice as sd

    stream = None
    wf = None

    def get_ts():
        return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

    log_raw = config["logs"]["raw_ar"]

    def log_to_file(text, *, path=log_raw):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except:
            pass

    # Load NVIDIA NeMo Model (RNNT Mode)
    try:
        print(f"[{get_ts()}] Loading NVIDIA FastConformer (RNNT)...")
        asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_name="nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0"
        ).cuda()

        # Switch to RNNT decoder for better boundary handling
        asr_model.change_decoding_strategy(decoder_type="rnnt")
        asr_model.eval()

        # Disable dithering/padding for stability
        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0

        print(f"[{get_ts()}] Model Loaded (RNNT Mode).")
    except Exception as e:
        event_q.put(("error", f"Model Load Error: {e}"))
        return

    # Audio Source Setup (Strict 16kHz)
    is_live_mic = isinstance(source, int)
    mic_queue = queue.Queue()
    TARGET_SR = 16000

    def mic_callback(indata, frames, time_info, status):
        if status:
            print(f"[MIC] {status}", flush=True)
        mic_queue.put(indata.copy())

    if is_live_mic:
        try:
            stream = sd.InputStream(
                device=source,
                channels=1,
                samplerate=TARGET_SR,
                dtype="float32",
                blocksize=int(TARGET_SR * 0.04),  # 40 ms
                callback=mic_callback,
            )
            stream.start()
        except Exception as e:
            event_q.put(("error", f"Mic Error: {e}"))
            return
    else:
        try:
            wf = wave.open(source, "rb")
            file_sr = wf.getframerate()
            if file_sr != TARGET_SR:
                print(f"Warning: File is {file_sr}Hz. Real-time resampling not implemented here.")
        except Exception as e:
            event_q.put(("error", f"File Error: {e}"))
            return

    # VAD & Buffers
    try:
        import webrtcvad
        vad = webrtcvad.Vad(1)  # Mode 1 (less aggressive, better for long vowels)
    except Exception:
        event_q.put(("error", "webrtcvad not installed."))
        return

    PROCESS_CHUNK_SEC = 6.0
    PROCESS_SAMPLES = int(TARGET_SR * PROCESS_CHUNK_SEC)
    VAD_FRAME_MS = 30
    VAD_FRAME_SAMPLES = int(TARGET_SR * (VAD_FRAME_MS / 1000))

    OVERLAP_SEC = 0.5
    OVERLAP_SAMPLES = int(TARGET_SR * OVERLAP_SEC)
    prev_audio_tail = np.array([], dtype=np.float32)

    audio_buffer = []
    silence_counter = 0
    MAX_TRAILING_SILENCE = int(TARGET_SR * 1.0)  # hold 1s of trailing silence
    MIN_TRANSCRIPTION_LEN = int(TARGET_SR * 3.0)  # require >= 3s of audio before flush
    MAX_BUFFER_LEN = int(TARGET_SR * 10.0)  # hard cap to avoid runaway latency

    batch_buf = []
    batch_start_id = None
    batch_end_id = None
    chunk_counter = 0
    refine_every = int(config.get("refine_every", 4))

    def flush_refine_batch(force_final: bool = False):
        nonlocal batch_buf, batch_start_id, batch_end_id
        if not batch_buf:
            return
        txt = "\n".join(batch_buf).strip()
        if not txt:
            return
        job = {
            "id": batch_end_id or batch_start_id or 0,
            "range": (batch_start_id or 0, batch_end_id or batch_start_id or 0),
            "source_ar": txt,
            "ts": time.time(),
            "log_file_ar": config["logs"].get("fixed_ar"),
            "log_file_en": config["logs"].get("final_en"),
            "log_file_de": config["logs"].get("final_de"),
            "final": bool(force_final),
        }
        refine_input_q.put(job)
        batch_buf, batch_start_id, batch_end_id = [], None, None

    def transcribe_segment(audio_np: np.ndarray) -> str:
        """Save audio to temp WAV and run RNNT inference."""
        if len(audio_np) < 800:
            return ""

        mx = np.max(np.abs(audio_np))
        if mx > 0:
            audio_np = audio_np / mx

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio_np, TARGET_SR)
            try:
                hyps = asr_model.transcribe([tmp.name], batch_size=1, return_hypotheses=False)
            except Exception as e:
                print(f"Transcribe Error: {e}")
                return ""

        if not hyps:
            return ""

        res = hyps[0]
        if isinstance(res, str):
            return res
        if hasattr(res, "text"):
            return res.text
        if isinstance(res, dict) and "text" in res:
            return res["text"]
        return str(res)

    print(f"[{get_ts()}] Stream Started (RNNT).")

    # Main Loop
    while not stop_event.is_set():
        # Read audio
        if is_live_mic:
            try:
                frames_list = []
                while True:
                    try:
                        frames_list.append(mic_queue.get_nowait())
                    except queue.Empty:
                        break
                if not frames_list:
                    time.sleep(0.01)
                    continue
                frame_np = np.concatenate(frames_list).flatten()
            except Exception:
                continue
        else:
            raw = wf.readframes(VAD_FRAME_SAMPLES)
            if not raw:
                break
            frame_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            time.sleep(0.01)  # 10ms throttle per 30ms frame in file mode

        if len(frame_np) == 0:
            continue

        # VAD & Buffer
        idx = 0
        while idx < len(frame_np):
            chunk = frame_np[idx : idx + VAD_FRAME_SAMPLES]
            idx += VAD_FRAME_SAMPLES
            if len(chunk) < VAD_FRAME_SAMPLES:
                continue

            int16_bytes = (chunk * 32767).astype(np.int16).tobytes()
            try:
                is_speech = vad.is_speech(int16_bytes, TARGET_SR)
            except Exception:
                is_speech = True

            if is_speech:
                audio_buffer.extend(chunk)
                silence_counter = 0
            else:
                silence_counter += len(chunk)
                # keep short trailing silence to bridge gaps
                if silence_counter < MAX_TRAILING_SILENCE:
                    audio_buffer.extend(chunk)

        # Trigger transcription
        do_transcribe = False
        buffer_len = len(audio_buffer)

        # Force flush if buffer is huge
        if buffer_len >= MAX_BUFFER_LEN:
            do_transcribe = True
        # Flush on silence only if we have enough speech
        elif silence_counter > MAX_TRAILING_SILENCE and buffer_len > MIN_TRANSCRIPTION_LEN:
            do_transcribe = True

        if do_transcribe:
            t_start = time.time()

            current_audio = np.array(audio_buffer, dtype=np.float32)
            if len(prev_audio_tail) > 0:
                full_input = np.concatenate([prev_audio_tail, current_audio])
            else:
                full_input = current_audio

            if len(current_audio) > OVERLAP_SAMPLES:
                prev_audio_tail = current_audio[-OVERLAP_SAMPLES:]
            else:
                prev_audio_tail = current_audio

            audio_buffer = []

            text = transcribe_segment(full_input)
            t_dur = time.time() - t_start

            clean = text.strip() if text else ""
            if clean:
                chunk_counter += 1
                log_msg = f"[{get_ts()}] [{chunk_counter}] (Infer: {t_dur:.2f}s) {clean}"
                print(log_msg)
                log_to_file(log_msg)

                event_q.put(("update", {"id": chunk_counter, "ar": clean, "infer_s": t_dur}))

                if batch_start_id is None:
                    batch_start_id = chunk_counter
                batch_end_id = chunk_counter
                batch_buf.append(clean)
                if len(batch_buf) >= refine_every:
                    flush_refine_batch()

    flush_refine_batch(force_final=True)
    event_q.put(("status", "stream_finished"))
    if is_live_mic and stream:
        stream.stop()
        stream.close()
