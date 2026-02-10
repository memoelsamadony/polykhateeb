# PolyKhateeb

**Real-time Arabic Islamic sermon (khutbah) transcription and translation system.**

PolyKhateeb ("poly" for multilingual + "khateeb" for sermon speaker) is an AI-powered system that listens to Arabic Islamic sermons in real-time and provides:
- Live Arabic speech recognition (ASR) using NVIDIA NeMo FastConformer
- Automatic correction of ASR errors using LLMs with Islamic glossary protection
- Real-time translation to English and German
- Hallucination filtering for Arabic religious text (ratio checks, boilerplate rejection)
- TV teleprompter mode with per-language viewers for display on smart TVs or projectors
- Telegram broadcasting for remote audiences
- Groq cloud polish with automatic model fallback on rate limits

## The Story

This project was born from a real need: making Islamic sermons accessible to multilingual communities in real-time. What started as a simple Whisper-based transcription experiment evolved through multiple iterations into a sophisticated multi-stage pipeline.

### Evolution (see `legacy/` folder for historical versions)

1. **Test1-2**: Initial experiments with Faster-Whisper + Gemini/MarianMT translation
2. **Test3-4**: Added local LLM support via OLLAMA with Qwen models for faster processing
3. **Test5-6**: Integrated Groq cloud API for high-quality translation, added dialect support and Arabic GEC
4. **Test7**: Consolidated monolith — NeMo FastConformer + OLLAMA refinement + Groq cloud polish + Telegram, all in one file
5. **Current (`src/`)**: Restructured into modular packages with shared state, TV teleprompter viewer, and per-language display pages

### Key Insights from Development

- **ASR hallucination is a real problem** - Arabic religious text is particularly challenging due to classical vocabulary and Quranic quotes
- **Context matters** - Sliding window of previous transcriptions dramatically improves LLM correction quality
- **Hybrid approach works best** - Fast local LLM (OLLAMA) for immediate results + cloud (Groq) for polished final output
- **Specialized models win** - NVIDIA's Arabic FastConformer outperforms general Whisper for khutbah speech

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌────────────┐
│ Audio Input │────▶│ NeMo ASR     │────▶│ Refinement      │────▶│ Streamlit  │
│ (Mic/File)  │     │ FastConformer│     │ Worker (OLLAMA) │     │ UI         │
└─────────────┘     └──────────────┘     └────────┬────────┘     └─────┬──────┘
                                                  │                    │
                                                  ▼                    ▼
                                         ┌─────────────────┐   ┌──────────────┐
                                         │ Cloud Polish    │   │ Shared State │
                                         │ Worker (Groq)   │   └──────┬───────┘
                                         └─────────────────┘          │
                                                                      ▼
                                         ┌─────────────────┐   ┌──────────────┐
                                         │ Telegram        │   │ TV Viewer    │
                                         │ Broadcast       │   │ (per-lang)   │
                                         └─────────────────┘   └──────────────┘

All three stages (ASR, Refinement, Cloud) feed into Telegram via the main UI.
Shared State enables multiple TV viewer sessions to read from one transcription.
```

### Components

#### Workers
- **src/workers/transcription.py**: NVIDIA NeMo FastConformer with VAD-based segmentation
- **src/workers/refinement.py**: Local LLM (OLLAMA) for ASR error correction + translation to EN/DE
- **src/workers/cloud_polish.py**: Groq cloud API for high-quality multi-language translation with model fallback

#### API Clients
- **src/api/ollama.py**: OLLAMA/RunPod LLM client with retry logic, `<think>` tag stripping, and Chinese hallucination guard
- **src/api/groq_client.py**: Groq client with per-model cooldown tracking and `retry-after` header handling
- **src/api/telegram.py**: Telegram message broadcaster with rate-limit handling and message splitting

#### UI
- **src/ui/streamlit_app.py**: Main control dashboard with 5-column live display
- **src/ui/shared_state.py**: Thread-safe process-level singleton for cross-session data sharing
- **src/ui/tv_viewer.py**: TV teleprompter viewer — hub page with per-language animated viewers

#### Core Utilities
- **src/config.py**: Environment loading, API keys, Groq fallback model chain, Islamic glossary, worker settings
- **src/utils.py**: Timestamped logging, log file path management, file append
- **src/arabic_utils.py**: Arabic ratio detection, hallucination filtering, glossary term protection/restoration, string similarity

## Requirements

- **Hardware**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- **OS**: Linux (Ubuntu 20.04+ tested)
- **Python**: 3.10+
- **FFmpeg**: For audio extraction from video files

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y ffmpeg

# For NVIDIA NeMo
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install nemo_toolkit['asr']
```

## Installation

```bash
# Clone the repository
git clone https://github.com/memoelsamadony/polykhateeb.git
cd polykhateeb

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```bash
# Groq API for cloud translation (get key from https://console.groq.com)
GROQ_API_KEY=your_groq_api_key

# Telegram bot for broadcasting (optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Local LLM API endpoint
# Default points to a RunPod proxy; override for local OLLAMA:
# OLLAMA_API_URL=http://localhost:11434/api/chat
OLLAMA_API_URL=https://your-runpod-endpoint/llm/chat
LLM_API_KEY=optional_api_key
```

## Usage

```bash
# Activate environment
source .venv/bin/activate

# Run the application
streamlit run main.py
```

### Control Dashboard

1. **Select audio source**: Microphone or file upload
2. **Configure hardware**: Choose compute device and model size
3. **Click START STREAM** to begin transcription
4. Watch live updates in the 5-column display:
   - Raw Arabic ASR output
   - Corrected Arabic
   - English translation
   - German translation
   - Cloud-polished translations

### TV Viewer (teleprompter mode)

Open on a smart TV, projector, or second device. The hub page links to per-language viewers:

| Page | URL |
|------|-----|
| Hub (all links) | `http://<laptop-ip>:8501/?mode=tv` |
| English | `http://<laptop-ip>:8501/?mode=tv&lang=en` |
| Deutsch | `http://<laptop-ip>:8501/?mode=tv&lang=de` |
| Arabic (refined) | `http://<laptop-ip>:8501/?mode=tv&lang=ar` |
| Arabic (raw ASR) | `http://<laptop-ip>:8501/?mode=tv&lang=raw` |

Each language page shows animated text blocks that fade in as new transcription arrives, with older blocks gradually dimming. The pages auto-refresh every 0.5 seconds and auto-scroll to the latest content.

## Project Structure

```
polykhateeb/
├── main.py                # Entry point
├── src/
│   ├── config.py          # Configuration, env vars, glossary, model fallback chain
│   ├── utils.py           # Logging, file path helpers
│   ├── arabic_utils.py    # Arabic ratio, hallucination filter, term protection
│   ├── api/
│   │   ├── ollama.py      # OLLAMA/RunPod LLM client with retry and response parsing
│   │   ├── groq_client.py # Groq client with per-model cooldown and fallback
│   │   └── telegram.py    # Telegram broadcaster with rate-limit handling
│   ├── workers/
│   │   ├── transcription.py # NeMo FastConformer ASR with VAD segmentation
│   │   ├── refinement.py    # Local LLM correction + EN/DE translation
│   │   └── cloud_polish.py  # Groq cloud batch translation with model fallback
│   └── ui/
│       ├── streamlit_app.py # Main control dashboard (5-column live display)
│       ├── shared_state.py  # Thread-safe cross-session state singleton
│       └── tv_viewer.py     # TV teleprompter hub + per-language viewers
├── legacy/                # Historical test versions (Test1-Test7)
├── experimental/          # Experimental scripts
├── ijaza/                 # Quran validation library (separate package)
├── logs/                  # Transcription logs (per-session, per-stage)
└── docs/                  # Documentation
```

## Related Projects

- **[Ijaza](https://github.com/memoelsamadony/ijaza)**: Quran verse validation library (developed as part of this project)

## License

MIT

## Acknowledgments

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) for the Arabic FastConformer model
- [Groq](https://groq.com) for fast cloud LLM inference
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) for the initial transcription experiments
- The open-source Islamic AI community
