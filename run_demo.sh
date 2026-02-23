#!/usr/bin/env bash
# Run the Pipecat voice demo (STT -> LLM -> TTS).
# Requires: DEEPGRAM_API_KEY, OPENAI_API_KEY (or use ./run_demo_with_record.sh for free models).
# With -t webrtc: open the printed URL in your browser to use laptop mic and speaker.

set -e
cd "$(dirname "$0")"

if [ -z "$DEEPGRAM_API_KEY" ] || [ -z "$OPENAI_API_KEY" ]; then
  echo "Set DEEPGRAM_API_KEY and OPENAI_API_KEY (or add them to .env and use python-dotenv)."
  echo "For no API keys, use: ./run_demo_with_record.sh"
  exit 1
fi

echo "Starting voice demo (STT=Deepgram, LLM=OpenAI, TTS=Qwen3-TTS)..."
echo "Open the URL in your browser and allow mic to speak; you'll hear the reply from your speaker."
# Default to webrtc so you get a local URL (laptop mic + speaker)
PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null)
exec "$PYTHON" demo_pipecat.py -t webrtc "$@"
