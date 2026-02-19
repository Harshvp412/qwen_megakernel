#!/usr/bin/env bash
# Run the Pipecat voice demo (STT -> LLM -> TTS).
# Requires: DEEPGRAM_API_KEY, OPENAI_API_KEY, and pip packages qwen-tts, pipecat-ai, websockets.

set -e
cd "$(dirname "$0")"

if [ -z "$DEEPGRAM_API_KEY" ] || [ -z "$OPENAI_API_KEY" ]; then
  echo "Set DEEPGRAM_API_KEY and OPENAI_API_KEY (or add them to .env and use python-dotenv)."
  echo "Example: export DEEPGRAM_API_KEY=... OPENAI_API_KEY=..."
  exit 1
fi

echo "Starting voice demo (STT=Deepgram, LLM=OpenAI, TTS=Megakernel-backed Qwen3-TTS)..."
exec python demo_pipecat.py "$@"
