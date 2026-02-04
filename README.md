# PolyKhateeb

**Real-time Arabic Islamic sermon (khutbah) transcription and translation system.**

PolyKhateeb ("poly" for multilingual + "khateeb" for sermon speaker) is an AI-powered system that listens to Arabic Islamic sermons in real-time and provides:
- Live Arabic speech recognition (ASR)
- Automatic correction of ASR errors using LLMs
- Real-time translation to English and German
- Telegram broadcasting for remote audiences

## The Story

This project was born from a real need: making Islamic sermons accessible to multilingual communities in real-time. What started as a simple Whisper-based transcription experiment evolved through multiple iterations into a sophisticated multi-stage pipeline.

### Evolution (see `legacy/` folder for historical versions)

1. **Test1-2**: Initial experiments with Faster-Whisper + Gemini/MarianMT translation
2. **Test3-4**: Added local LLM support via OLLAMA with Qwen models for faster processing
3. **Test5-6**: Integrated Groq cloud API for high-quality translation, added dialect support
4. **Current**: NVIDIA NeMo FastConformer for Arabic ASR + multi-tier LLM pipeline

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
└─────────────┘     └──────────────┘     └────────┬────────┘     └────────────┘
                                                  │
                                                  ▼
                                         ┌─────────────────┐     ┌────────────┐
                                         │ Cloud Polish    │────▶│ Telegram   │
                                         │ Worker (Groq)   │     │ Broadcast  │
                                         └─────────────────┘     └────────────┘
```

### Components

- **src/workers/transcription.py**: NVIDIA NeMo FastConformer with VAD-based segmentation
- **src/workers/refinement.py**: Local LLM (OLLAMA) for ASR error correction + translation
- **src/workers/cloud_polish.py**: Groq cloud API for high-quality multi-language translation
- **src/api/telegram.py**: Real-time Telegram message broadcasting
- **src/ui/streamlit_app.py**: Web interface with live updating columns

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

# Local LLM API (if using RunPod or similar)
OLLAMA_API_URL=http://localhost:11434/api/chat
LLM_API_KEY=optional_api_key
```

## Usage

```bash
# Activate environment
source .venv/bin/activate

# Run the application
streamlit run main.py
```

### In the UI

1. **Select audio source**: Microphone or file upload
2. **Configure hardware**: Choose compute device and model size
3. **Click START STREAM** to begin transcription
4. Watch live updates in the 5-column display:
   - Raw Arabic ASR output
   - Corrected Arabic
   - English translation
   - German translation
   - Cloud-polished translations

## Project Structure

```
polykhateeb/
├── main.py              # Entry point
├── src/
│   ├── config.py        # Configuration and environment
│   ├── utils.py         # Utility functions
│   ├── arabic_utils.py  # Arabic text processing
│   ├── api/
│   │   ├── ollama.py    # Local LLM client
│   │   ├── groq_client.py # Groq cloud client
│   │   └── telegram.py  # Telegram broadcaster
│   ├── workers/
│   │   ├── transcription.py # NeMo ASR worker
│   │   ├── refinement.py    # Local LLM worker
│   │   └── cloud_polish.py  # Cloud translation worker
│   └── ui/
│       └── streamlit_app.py # Streamlit interface
├── legacy/              # Historical test versions (Test1-Test6)
├── experimental/        # Experimental scripts
├── ijaza/               # Quran validation library (separate package)
├── logs/                # Transcription logs
└── docs/                # Documentation
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
