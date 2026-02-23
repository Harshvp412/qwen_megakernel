#!/usr/bin/env bash
# Run the Pipecat voice demo with recording enabled.
# Saves conversation audio to recordings/demo_YYYYMMDD_HHMMSS.wav when the client disconnects.
#
# --- Option 1: FREE (no API keys) ---
# Uses local Whisper (STT) + Ollama (LLM). Install once:
#   pip install "pipecat-ai[whisper]" "pipecat-ai[ollama]"
#   ollama serve   # in another terminal if not already running
#   ollama pull llama2
# Then run:
#   ./run_demo_with_record.sh
#
# --- Option 2: With API keys ---
#   export DEEPGRAM_API_KEY=... OPENAI_API_KEY=...
#   ./run_demo_with_record.sh
#
# --- Reliable TTS when megakernel has CUDA issues ---
#   USE_LEGACY_TTS=1 ./run_demo_with_record.sh

set -e
cd "$(dirname "$0")"

# If no API keys, demo will use free models (Whisper + Ollama) automatically
if [ -z "$DEEPGRAM_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
  echo "No API keys set. Using FREE local models: Whisper (STT) + Ollama (LLM)."
  echo "  Prereqs: pip install \"pipecat-ai[whisper]\" \"pipecat-ai[ollama]\""
  echo "           ollama serve  (in another terminal)"
  echo "           ollama pull llama2"
  echo ""
  export USE_FREE_MODELS=1
fi

export RECORD_DEMO=1
# Optional: use legacy (HF) TTS when megakernel TTS crashes
# export USE_LEGACY_TTS=1

echo "Voice demo with recording (RECORD_DEMO=1)."
echo "  - Use WebRTC so you get a LOCAL URL (no Daily account)."
echo "  - Open the URL in your browser (e.g. http://localhost:7860/client), allow mic, speak."
echo "  - You hear the bot from your speaker. When done, disconnect."
echo "  - Recording saved to recordings/demo_YYYYMMDD_HHMMSS.wav"
echo ""
if [ -n "$USE_LEGACY_TTS" ]; then
  echo "  - USE_LEGACY_TTS=1: using HF-based TTS for reliable demo."
fi
echo ""

# -t webrtc = local URL, use laptop mic and speaker in browser
PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null)
exec "$PYTHON" demo_pipecat.py -t webrtc "$@"
