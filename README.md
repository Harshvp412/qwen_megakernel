## Qwen3-0.6B Megakernel for RTX 5090

Single-kernel decode for Qwen3-0.6B (bf16) — all 28 transformer layers fused
into one persistent CUDA kernel, with a separate LM-head kernel.
Optimised exclusively for **RTX 5090 (sm_120, CUDA 12.8+)**.

More details: https://blog.alpindale.net/posts/5090_decode_optimization/

| Backend      | tok/s  | ms/tok | Speedup |
|--------------|--------|--------|---------|
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

### Credits

Based on Elliot Arledge's [MegaQwen](https://github.com/Infatoshi/MegaQwen)
for the RTX 3090.
Original kernel: [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel)
