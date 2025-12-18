# Realtime Transcribe & Translate (Streamlit + Faster-Whisper + Gemini)

This app lets you:
- Capture microphone audio (via `streamlit-webrtc`) and get realtime transcription.
- Upload an audio or video file to simulate realtime processing.
- Translate the transcribed text to English or German using Google Gemini.

## Requirements
- Linux with `bash` shell
- Python 3.10+
- `ffmpeg` installed on the system (required by Faster-Whisper)

Install ffmpeg on Ubuntu/Debian:
```bash
sudo apt update
sudo apt install -y ffmpeg
```

## Setup (venv)
From the project root `RealTimeTranslation`:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configure Gemini API Key
Set your key as an environment variable, or paste it in the sidebar:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

## Run the App
```bash
source .venv/bin/activate
streamlit run buff.py
```

## Usage
- In the sidebar, choose input source: Microphone, Audio File, or Video File.
- Choose target language (English/German).
- For microphone, allow browser access. Partial transcripts appear every ~5 seconds.
- For audio/video, upload a file; the app will transcribe and translate.

## Notes
- Model used: Faster-Whisper `large-v2` (downloads on first run).
- Translation model: `models/gemini-2.5-flash`.
- If you experience performance issues on CPU, consider using a smaller Whisper model (e.g., `medium`, `small`).
