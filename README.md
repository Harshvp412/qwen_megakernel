## Qwen3-0.6B Megakernel for RTX 5090

Single-kernel decode for Qwen3-0.6B (bf16) — all 28 transformer layers fused
into one persistent CUDA kernel, with a separate LM-head kernel.
Optimised exclusively for **RTX 5090 (sm_120, CUDA 12.8+)**.

More details: https://blog.alpindale.net/posts/5090_decode_optimization/

**Assignment alignment:** This repo wires the megakernel into a Pipecat voice pipeline using Qwen3-TTS for audio. For a strict requirement-by-requirement checklist (what’s done vs what’s missing — e.g. megakernel as talker decoder, streaming, TTFC/RTF targets), see **[ASSIGNMENT_CHECKLIST.md](ASSIGNMENT_CHECKLIST.md)**.

| Backend      | tok/s  | ms/tok | Speedup |
| ------------ | ------ | ------ | ------- |
| PyTorch (HF) | 123.3  | 8.11   | 1.00x   |
| Megakernel   | 1036.3 | 0.99   | 8.40x   |

---

### Requirements

- GPU: NVIDIA RTX 5090 (sm_120). Will not run on other GPUs.
- CUDA: 12.8 or newer
- Python: 3.10+
- OS: Linux (tested), macOS CPU/MPS (parity script only)

---

### GPU Setup (RTX 5090 machine)

```bash
# 1. Clone
git clone git@github.com:Harshvp412/qwen_megakernel.git
cd qwen_megakernel

# 2. Install PyTorch with CUDA 12.8 support
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Build the CUDA kernel  (~1–2 min first time, cached after)
#    The kernel compiles on first import via __init__.py → build.py → get_extension()
python -c "import qwen_megakernel; print('Build OK')"

# 5. Run benchmark
python -m qwen_megakernel.bench
```

---

### Parity Check (verify kernel output matches HuggingFace)

The ground-truth token sequence is pre-generated and committed.
On the GPU machine, just run:

```bash
python compare_tokens.py
```

Expected output:

```
pos  ref_id   mk_id   ref_text              mk_text               match
─────────────────────────────────────────────────────────────────────────
  0   12095   12095   ' Paris'              ' Paris'              ✓
  1      13      13   '.'                   '.'                   ✓
 ...
─────────────────────────────────────────────────────────────────────────
══════════════════════════════════════
  Parity: TRUE  ✓  All tokens match
══════════════════════════════════════
```

To regenerate the reference (e.g. after changing the prompt or model):

```bash
# Mac or any CPU machine
python parity_reference.py --model Qwen/Qwen3-0.6B
```

---

### Key Implementation Notes

**RoPE theta**
Qwen3-0.6B uses `rope_theta = 1_000_000` (stored in
`config.rope_scaling['rope_theta']`), not the legacy 10 000.
`load_weights()` reads this from the model config at runtime — no hardcoded constant.

**LM head weight**
`lm_head.weight` is loaded explicitly from the state dict (not aliased to
`embed_tokens.weight`). transformers 5.x stores both as separate tensors even
when `tie_word_embeddings=True`. A shape assertion fires at load time if there
is a mismatch.

**Kernel target**

```
sm_120  (Blackwell — RTX 5090 only)
LDG_NUM_BLOCKS = 128, LDG_BLOCK_SIZE = 512
```

---

### TTS Integration (Qwen3-TTS-12Hz-0.6B-Base)

```bash
pip install qwen-tts
python parity_reference.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base
python compare_tokens.py --ref parity_reference_output.json
```

Config mismatches (vocab size, rope_theta, untied lm_head) are reported
automatically at load time by `parity_reference.py`.

---

### Pipeline & voice integration (Steps 2–4)

The repo wires the megakernel into a voice pipeline:

- **Step 2** — `inference_server.py`: `MegakernelDecoder` (streaming token generation). TTS: `MegakernelTalkerBackend` (megakernel as talker → codec → audio) preferred; `Qwen3TTSTalkerBackend` (qwen-tts end-to-end) fallback.
- **Step 3** — `pipecat_tts_service.py`: `Qwen3TTSPipecatService` (Pipecat TTS; uses megakernel-as-talker backend by default).
- **Step 4** — Full pipeline test and docs: run all steps and check expected results.

**Architecture**

```
Text → MegakernelDecoder (streaming tokens)
     → MegakernelTalkerBackend (megakernel talker → codec tokens → codec/vocoder → audio chunks)
     → Qwen3TTSPipecatService (Pipecat frames)
     → demo_pipecat: STT (Deepgram) → LLM (OpenAI) → TTS (Qwen3) → audio out
```

**Run tests (GPU machine)**

```bash
# Step 2: Inference server (streaming + benchmark; TTS optional)
python test_step2_inference_server.py

# Megakernel-as-talker TTS backend (standalone test; achieves RTF < 0.3)
python test_megakernel_tts_backend.py
python test_megakernel_tts_backend.py --wav /tmp/out.wav --text "Hello world."
# Reported RTF with this path is below the 0.3 target (e.g. ~0.23 in testing).

# Step 3: Pipecat TTS service (optional if pipecat-ai / qwen-tts not installed)
python test_step3_pipecat.py

# Step 4: Full pipeline — all runnable checks with expected results
python test_step4_pipeline.py
```

**Expected results (Step 4)**

| Check                | Expected                                                                       |
| -------------------- | ------------------------------------------------------------------------------ |
| Step 2.1 Streaming   | PASS (tokens stream correctly)                                                 |
| Step 2.2 TTS Backend | PASS when audio generated; **RTF < 0.3** (e.g. ~0.23). **TTFC < 90 ms:** run `python profile_codec_decode.py` then `python test_megakernel_tts_backend.py --first-chunk-frames 1` (or 2) so first codec decode is small. |
| Step 2.3 Tok/s       | ≥ 500 (target ~1000)                                                           |
| Step 3 Pipecat TTS   | PASS (if pipecat + qwen-tts)                                                   |

Step 2.2 and Step 3 are **optional** (SKIP if qwen-tts or pipecat not installed). Step 4 passes when every _runnable_ check meets the above.

**Voice demo (STT → LLM → TTS)**

Requires `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, and `qwen-tts` (+ `websockets`). Optional: `python-dotenv` and a `.env` file.

```bash
pip install qwen-tts websockets pipecat-ai python-dotenv
python demo_pipecat.py
# Or: python demo_pipecat.py -t webrtc
```

Open the URL printed (e.g. WebRTC client), connect, and talk. Pipeline: Deepgram STT → OpenAI LLM → Qwen3-TTS (our service) → audio out.

**Optional system dependencies (for TTS)**

- **SoX** — Used by qwen-tts (or its dependencies) for audio processing. The pipeline works without it (you may see “SoX could not be found”); to silence the warning and enable any SoX-based features:
  ```bash
  # Debian/Ubuntu
  sudo apt-get update && sudo apt-get install -y sox
  ```
- **flash-attn** — Optional; qwen-tts will warn “flash-attn is not installed” and use a slower path. Install only if system CUDA matches the CUDA version PyTorch was built with (e.g. PyTorch built for 12.8 needs system CUDA 12.8). If you see “CUDA version mismatches” (e.g. system 13.0 vs PyTorch 12.8), skip flash-attn; TTS runs without it.

**Known limitations**

- **Parity:** On the same machine, HF `generate()` and megakernel match (see `compare_hf_generate_vs_mk.py`). Regenerate `parity_reference_output.json` on the target machine for `compare_tokens.py`.
- **transformers version:** Pinned to `4.57.3` so `qwen-tts` installs in the same env; model loading uses `torch_dtype=` for compatibility.
- **Streaming TTS:** Current TTS path generates full utterance then chunks; true streaming (token-by-token to audio) is not wired yet.

---

### Credits

Based on Elliot Arledge's [MegaQwen](https://github.com/Infatoshi/MegaQwen)
for the RTX 3090.
Original kernel: [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel)
