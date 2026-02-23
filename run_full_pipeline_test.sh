#!/bin/bash
# Full pipeline test: complete install + run Step 4 and optional demo.
# Use on GPU machine (e.g. RTX 5090) with CUDA 12.8+.
# Optional: set DEEPGRAM_API_KEY and OPENAI_API_KEY for voice demo.

set -e
cd "$(dirname "$0")"

echo "=============================================="
echo "1. Pull latest"
echo "=============================================="
git pull origin master || true

echo ""
echo "=============================================="
echo "2. Install PyTorch (CUDA 12.8) if needed"
echo "=============================================="
pip install torch --index-url https://download.pytorch.org/whl/cu128 2>/dev/null || echo "(PyTorch already installed?)"

echo ""
echo "=============================================="
echo "3. Install core dependencies"
echo "=============================================="
pip install -r requirements.txt

echo ""
echo "=============================================="
echo "4. Build CUDA kernel (first run may take 1–2 min)"
echo "=============================================="
PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null)
"$PYTHON" -c "import qwen_megakernel; print('Build OK')"

echo ""
echo "=============================================="
echo "5. Run Step 4 – full pipeline validation"
echo "=============================================="
"$PYTHON" tests/test_step4_pipeline.py
STEP4_EXIT=$?

if [ $STEP4_EXIT -ne 0 ]; then
  echo ""
  echo "Step 4 had failures. Fix errors above or install optional deps:"
  echo "  pip install qwen-tts   # for TTS backend + Pipecat TTS tests"
  echo "  pip install websockets   # if pipecat needs it"
  exit $STEP4_EXIT
fi

echo ""
echo "=============================================="
echo "6. Optional: voice demo (needs API keys)"
echo "=============================================="
if [ -n "${DEEPGRAM_API_KEY}" ] && [ -n "${OPENAI_API_KEY}" ]; then
  echo "API keys set. Start demo with: $PYTHON demo_pipecat.py"
  echo "Then open the printed URL in a browser and connect."
else
  echo "Set DEEPGRAM_API_KEY and OPENAI_API_KEY to run: $PYTHON demo_pipecat.py"
fi

echo ""
echo "=============================================="
echo "Full pipeline test complete."
echo "=============================================="
