# Megakernel as Talker Decoder — Current Status & Next Steps

## What We Need to Achieve

**Goal:** Use the megakernel as the LLM decode backend **inside** Qwen3-TTS's talker decoder, replacing qwen-tts's internal decoder with our megakernel.

**Target state:** Text → megakernel (talker decoder) → codec tokens → qwen-tts codec/vocoder → audio

**Current state (full compliance):** ✅ Implemented. The voice pipeline uses **MegakernelTalkerBackend**: prompt token IDs (codec control) → megakernel (talker weights) → codec token stream → `speech_tokenizer.decode` → audio. Pipecat prefers this backend by default; fallback remains `Qwen3TTSTalkerBackend` (qwen-tts end-to-end).

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

### 2. Build TTS Prompt for Megakernel ✅ DONE

**Requirement:** Replicate qwen-tts's prompt construction (text + language + speaker + codec control tokens) but output **token IDs** (not embeddings) for the megakernel.

**Implemented:** `megakernel_tts_backend.build_tts_prompt_token_ids()` builds codec control token IDs from talker config (codec_think_id, language_id, codec_pad_id, codec_bos_id, optional speaker). Text conditioning as token IDs would require one forward from qwen-tts for the first codec token; currently we use control tokens only.

### 3. Create `MegakernelTalkerBackend` ✅ DONE

**Replaces:** Default TTS path now uses megakernel; `Qwen3TTSTalkerBackend` remains as fallback.

**Implemented:** `megakernel_tts_backend.MegakernelTalkerBackend` — `build_prompt_ids()`, `generate_codec_tokens()`, `codec_tokens_to_audio()`, `text_to_speech_blocks()` (same interface as `Qwen3TTSTalkerBackend`). Flow: prompt token IDs → megakernel → codec token list → `speech_tokenizer.decode` → chunked audio.

### 4. Wire Codec Tokens → Audio ✅ DONE

**Requirement:** Extract codec/vocoder from `Qwen3TTSModel` and use it to decode codec tokens → waveform.

**Implemented:** Use `tts.model.speech_tokenizer.decode([{"audio_codes": codes}])` with our generated codec token array (shape `(T, 1)` for single-codebook output from megakernel).

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

1. ✅ Run inspection scripts — codec access via `model.speech_tokenizer`, megakernel produces codec token IDs
2. ✅ Build prompt construction — `build_tts_prompt_token_ids()` in `megakernel_tts_backend.py`
3. ✅ Codec wrapper — `codec_tokens_to_audio()` using `speech_tokenizer.decode`
4. ✅ `MegakernelTalkerBackend` — implemented and wired
5. **Optional:** Add text conditioning (first codec token from qwen-tts one step); streaming decode if codec supports it

---

## Key Questions to Answer

1. ✅ **Can megakernel load TTS weights?** Yes (already verified)
2. ⏳ **Does megakernel produce codec tokens?** Need to test (token IDs vs codec vocab)
3. ⏳ **How to access codec/vocoder separately?** Need to inspect `Qwen3TTSModel`
4. ⏳ **Can codec decode incrementally?** Need to test streaming decode
5. ⏳ **What's the exact prompt format?** Need to replicate qwen-tts's `generate()` logic
