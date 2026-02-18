# Qwen3-TTS + Pipecat Integration Plan

## Overview
Integrate AlpinDale's qwen_megakernel as the LLM decode backend for Qwen3-TTS talker decoder, streaming into Pipecat voice pipeline.

## Current Status
- ✅ Megakernel working for Qwen3-0.6B (parity issue documented, but functional)
- ✅ Python wrapper (`Decoder` class) ready
- ⏳ TTS model compatibility check needed
- ⏳ Inference server wrapper needed
- ⏳ Pipecat integration needed

## Step-by-Step Implementation

### Step 1: Verify TTS Model Compatibility ✅ IN PROGRESS

**Task 1.1: Inspect TTS Architecture**
- Run `python inspect_tts_model.py` to check:
  - Vocab size (must match 151936)
  - Hidden size (must match 1024)
  - Number of layers (must match 28)
  - Number of KV heads (must match 8)
  - Head dimension (must match 128)

**Task 1.2: Load TTS Weights**
- Update `model.py` to support TTS model paths
- Handle `qwen3_tts` architecture (may need `qwen-tts` package)
- Extract talker decoder weights (not codebook generator)

**Task 1.3: Validate TTS Parity**
- Create `parity_reference_tts.py` to generate HF reference
- Run `compare_tokens.py` with TTS model
- Document any differences

**Expected Output:** Confirmation that TTS talker decoder matches Qwen3-0.6B architecture

---

### Step 2: Build Inference Server

**Task 2.1: Streaming Token Generator**
- Create `inference_server.py` with:
  - `generate_stream(prompt: str) -> Iterator[int]` - yields tokens as generated
  - Handle TTS-specific decode loop (stop conditions, special tokens)
  - Thread-safe for concurrent requests

**Task 2.2: Audio Generation Pipeline**
- Integrate Qwen3-TTS codec/vocoder:
  - Token stream → audio codec → waveform
  - Use Qwen3-TTS's built-in audio generation
  - Stream audio chunks (not full utterance)

**Task 2.3: Performance Optimization**
- Pre-warm decoder (load weights once)
- Batch handling if needed
- Measure tok/s, latency

**Expected Output:** `InferenceServer` class that takes text → streams audio chunks

---

### Step 3: Pipecat Integration ✅ DONE

**Task 3.1: Custom TTS Service**
- Created `pipecat_tts_service.py`:
  - `Qwen3TTSPipecatService(TTSService)` with `run_tts(text, context_id) -> AsyncGenerator[Frame, None]`
  - Yields `TTSStartedFrame`, `TTSAudioRawFrame` (from backend chunks), `TTSStoppedFrame`
  - Uses `inference_server.Qwen3TTSTalkerBackend` (optional injectable)

**Task 3.2: Pipeline Setup**
- Created `demo_pipecat.py`:
  - STT: Deepgram, LLM: OpenAI, TTS: Qwen3TTSPipecatService
  - Pipeline: transport.input() → stt → user_aggregator → llm → tts → transport.output() → assistant_aggregator
  - Run: `python demo_pipecat.py` (or `-t webrtc`); requires DEEPGRAM_API_KEY, OPENAI_API_KEY, and qwen-tts

**Task 3.3: Streaming Validation**
- Verify frames stream frame-by-frame (not buffered)
- Check audio quality
- Measure TTFC (target < 90ms)

**Expected Output:** Working Pipecat pipeline with streaming TTS

---

### Step 4: Benchmarking & Documentation

**Task 4.1: Performance Metrics**
- Measure:
  - Tokens/sec from megakernel
  - TTFC (time to first audio chunk) - target < 90ms
  - RTF (real-time factor) - target < 0.3
  - End-to-end latency

**Task 4.2: Documentation**
- Update README.md with:
  - Architecture decisions
  - Kernel modifications (if any)
  - How to run demo
  - Performance numbers
  - Known limitations

**Task 4.3: Demo**
- Create demo script/recording
- Show voice agent working end-to-end

**Expected Output:** Complete documentation + performance report

---

## File Structure

```
qwen_megakernel/
├── qwen_megakernel/
│   ├── model.py              # ✅ Already supports Qwen3-0.6B
│   └── __init__.py
├── csrc/
│   └── kernel.cu             # ✅ CUDA kernel (no changes needed if TTS matches)
├── inference_server.py       # ⏳ NEW: Streaming inference server
├── pipecat_tts_service.py   # ⏳ NEW: Pipecat TTS service
├── demo_pipecat.py          # ⏳ NEW: End-to-end demo
├── inspect_tts_model.py      # ✅ NEW: TTS architecture checker
├── requirements.txt          # ⏳ Update with pipecat, qwen-tts
└── README.md                # ⏳ Update with TTS integration docs
```

---

## Dependencies to Add

```txt
pipecat-ai>=0.0.1
qwen-tts>=0.1.0
soundfile>=0.12.0  # Already added
```

---

## Performance Targets

- **TTFC (Time to First Chunk):** < 90ms
- **RTF (Real-Time Factor):** < 0.3 (1s audio in < 300ms)
- **Tokens/sec:** ~1000 (from megakernel benchmark)
- **Streaming:** Frame-by-frame, not buffered

---

## Next Immediate Actions

1. ✅ Created `inspect_tts_model.py` - run this to check compatibility
2. ⏳ Run inspection script on GPU machine
3. ⏳ If compatible: proceed to Step 2
4. ⏳ If incompatible: document differences and adjust kernel constants
