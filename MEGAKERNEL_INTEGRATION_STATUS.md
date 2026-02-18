# Megakernel as Talker Decoder — Current Status & Next Steps

## What We Need to Achieve

**Goal:** Use the megakernel as the LLM decode backend **inside** Qwen3-TTS's talker decoder, replacing qwen-tts's internal decoder with our megakernel.

**Current state:** We use qwen-tts end-to-end (its decoder + codec). The megakernel is used separately for standalone decode, not inside TTS.

**Target state:** Text → megakernel (talker decoder) → codec tokens → qwen-tts codec/vocoder → audio

---

## What's Required

### 1. Understand Qwen3-TTS Architecture ✅ IN PROGRESS

**Files created:**
- `inspect_qwen_tts_codec.py` — Inspect model structure, find codec/vocoder access
- `test_megakernel_tts_tokens.py` — Test if megakernel produces codec tokens with TTS weights

**Next:** Run these scripts on GPU to understand:
- How to access codec/vocoder separately from `Qwen3TTSModel`
- Whether megakernel produces codec tokens (vs text tokens) when loaded with TTS weights
- What the codec token format/vocab looks like

### 2. Build TTS Prompt for Megakernel

**Requirement:** Replicate qwen-tts's prompt construction (text + language + speaker + codec control tokens) but output **token IDs** (not embeddings) for the megakernel.

**From `modeling_qwen3_tts.py`:** The `generate()` method builds:
- Text embeddings (via `text_projection`)
- Codec prefill tokens (language ID, speaker ID, BOS)
- Special tokens (`<|im_start|>assistant\n`, etc.)

**Need:** Convert this to token ID sequence that megakernel can decode.

### 3. Create `MegakernelTalkerBackend`

**Replaces:** Current `Qwen3TTSTalkerBackend` (which uses qwen-tts end-to-end)

**Interface:**
```python
class MegakernelTalkerBackend:
    def text_to_speech_blocks(self, text, language="English", ref_audio=None, ref_text=None):
        # 1. Build prompt token IDs (text + codec control tokens)
        prompt_ids = self._build_tts_prompt(text, language, ...)
        # 2. Stream codec tokens from megakernel
        codec_tokens = []
        for token_id in self.megakernel_decoder.generate_token_ids(prompt_ids):
            codec_tokens.append(token_id)
            # 3. Decode codec tokens → audio (incrementally if possible)
            if len(codec_tokens) >= chunk_size:  # or decode each token
                audio_chunk = self.codec.decode(codec_tokens)
                yield audio_chunk, sample_rate
                codec_tokens = []
```

### 4. Wire Codec Tokens → Audio

**Requirement:** Extract codec/vocoder from `Qwen3TTSModel` and use it to decode codec tokens → waveform.

**Challenge:** Qwen3-TTS may not expose codec/vocoder separately. May need to:
- Access `model.speech_tokenizer` directly
- Or patch/monkey-patch `generate()` to intercept codec tokens before decoding
- Or reverse-engineer the codec decode API

### 5. Make It Streaming

**Requirement:** Decode codec tokens incrementally (not wait for full sequence).

**Approach:**
- Megakernel already streams token IDs ✅
- Need codec to decode incrementally (may require buffering small windows)

---

## Implementation Plan

### Phase 1: Discovery (Current)

1. ✅ Created inspection scripts
2. ⏳ Run `inspect_qwen_tts_codec.py` — understand codec access
3. ⏳ Run `test_megakernel_tts_tokens.py` — verify megakernel produces codec tokens
4. ⏳ Document findings: codec API, token format, streaming capability

### Phase 2: Prompt Construction

1. Extract qwen-tts prompt building logic
2. Convert to token ID format (not embeddings)
3. Test: build prompt, feed to megakernel, verify output

### Phase 3: Codec Integration

1. Create codec wrapper: `codec_tokens → audio`
2. Test: manual codec tokens → decode → verify audio quality
3. Test streaming: incremental decode if possible

### Phase 4: Full Integration

1. Create `MegakernelTalkerBackend`
2. Replace `Qwen3TTSTalkerBackend` in `pipecat_tts_service.py`
3. Test end-to-end: text → megakernel → codec tokens → audio → Pipecat

### Phase 5: Streaming & Performance

1. Optimize streaming (minimize buffering)
2. Measure TTFC/RTF (target <90ms, <0.3)
3. Benchmark vs current implementation

---

## Files Created

- `MEGAKERNEL_TTS_INTEGRATION_PLAN.md` — Detailed technical plan (3 options analyzed)
- `MEGAKERNEL_INTEGRATION_STATUS.md` — This file (status & next steps)
- `inspect_qwen_tts_codec.py` — Inspect qwen-tts model structure
- `test_megakernel_tts_tokens.py` — Test megakernel with TTS weights

---

## Next Immediate Actions

1. **Run inspection scripts on GPU** to understand codec access and token format
2. **Document findings** — codec API, whether megakernel produces codec tokens
3. **Build prompt construction** — replicate qwen-tts prompt as token IDs
4. **Create codec wrapper** — `codec_tokens → audio` interface
5. **Implement `MegakernelTalkerBackend`** — wire everything together

---

## Key Questions to Answer

1. ✅ **Can megakernel load TTS weights?** Yes (already verified)
2. ⏳ **Does megakernel produce codec tokens?** Need to test (token IDs vs codec vocab)
3. ⏳ **How to access codec/vocoder separately?** Need to inspect `Qwen3TTSModel`
4. ⏳ **Can codec decode incrementally?** Need to test streaming decode
5. ⏳ **What's the exact prompt format?** Need to replicate qwen-tts's `generate()` logic
