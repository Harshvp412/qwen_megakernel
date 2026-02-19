# Megakernel as Qwen3-TTS Talker Decoder — Integration Plan

## Current Architecture Analysis

From `qwen_tts/core/models/modeling_qwen3_tts.py`:

**Qwen3TTSForConditionalGeneration.generate()** flow:
1. Preprocess: text → text embeddings, speaker embeddings, language IDs
2. Build `talker_input_embeds` (combines text + codec + speaker embeddings)
3. Call `self.talker.generate(inputs_embeds=talker_input_embeds, ...)`
4. `talker.generate()` → `talker.forward()` → `talker.model(inputs_embeds=...)` → hidden states → `codec_head` → codec tokens
5. Codec tokens → `speech_tokenizer` (codec/vocoder) → audio

**Key components:**
- `self.talker.model` = `Qwen3TTSTalkerModel` (decoder layers: 28 layers, hidden_size=1024, vocab_size=codec vocab ~151936)
- `self.talker.codec_head` = `nn.Linear(hidden_size, vocab_size)` (hidden → codec token logits)
- `self.talker.code_predictor` = predicts additional codec codes (multi-codebook)

**Challenge:** The talker uses **embeddings** (text + codec + speaker combined), not token IDs. Our megakernel expects token IDs and produces token IDs.

---

## Integration Strategy

### Option A: Replace `talker.model` with Megakernel (Decoder Layers Only)

**What:** Replace `Qwen3TTSTalkerModel` (the 28 decoder layers) with our megakernel, but keep:
- Text embedding projection
- Codec head (hidden → codec tokens)
- Code predictor
- Speaker/language handling

**How:**
1. Create `MegakernelTalkerModel` wrapper that:
   - Takes `inputs_embeds` (from preprocessing)
   - Converts embeddings → token IDs (via nearest-neighbor lookup or learned projection)
   - Runs megakernel decoder (token IDs → hidden states)
   - Returns hidden states compatible with `codec_head`

**Problem:** Megakernel expects token IDs, but talker uses embeddings. We'd need embedding→token conversion which loses information.

---

### Option B: Replace Entire Talker Generation with Megakernel Token Stream

**What:** Use megakernel to generate codec tokens directly, bypassing the talker's decoder layers entirely.

**How:**
1. Extract/prepare text prompt (same as qwen-tts preprocessing)
2. Load talker decoder weights into megakernel (we already support TTS model path)
3. Run megakernel: text prompt → codec token stream (autoregressive)
4. Feed codec tokens → qwen-tts codec/vocoder → audio

**Requirements:**
- Megakernel must produce **codec tokens** (not text tokens)
- Need to map text → codec token sequence (or use talker's text→codec mapping)
- Need access to qwen-tts's codec/vocoder separately (not just `generate_voice_clone`)

**Problem:** The talker decoder's vocab is codec tokens (~151936), but the megakernel is currently set up for text tokens. We'd need to verify the megakernel can load talker weights and produce codec tokens correctly.

---

### Option C: Hybrid — Megakernel for Main Decoder, Keep Code Predictor

**What:** Replace `talker.model` (main decoder) with megakernel, but keep `code_predictor` and embedding handling.

**Flow:**
1. Preprocess text → `talker_input_embeds` (as qwen-tts does)
2. **Replace `talker.model(inputs_embeds)` call** with:
   - Convert `inputs_embeds` → approximate token IDs (or use a learned embedding→token projection)
   - Run megakernel decoder (token IDs → hidden states)
   - Return hidden states
3. Continue: `codec_head(hidden_states)` → codec tokens
4. Code predictor → additional codec codes
5. Codec tokens → vocoder → audio

**Problem:** Still need embedding→token conversion. Also, the talker uses multimodal RoPE (3D position IDs for temporal/height/width), which our megakernel doesn't support.

---

## Recommended Approach: Option B (Simpler, More Direct)

**Why:** The assignment says "megakernel runs the talker decoder" — meaning the megakernel should produce the codec token stream that the talker normally produces. Option B is closest to that.

**Steps:**

### 1. Verify Megakernel Can Load Talker Weights and Produce Codec Tokens

- ✅ Already done: `model.py` supports TTS model path
- ✅ Already done: `inspect_tts_model.py` confirms shapes match
- ⚠️ Need to verify: megakernel produces codec tokens (not text tokens) when loaded with TTS weights

### 2. Create Text → Codec Token Prompt Mapping

The talker's `generate()` builds a complex prompt with:
- Text embeddings (via `text_projection`)
- Codec prefill tokens (language, speaker, BOS)
- Special tokens (`<|im_start|>assistant\n`, etc.)

We need to replicate this prompt construction but output **token IDs** (not embeddings) for the megakernel.

**Or:** Use qwen-tts's tokenizer to convert text → token IDs, then prepend codec control tokens.

### 3. Create `MegakernelTalkerBackend` That Produces Codec Token Stream

```python
class MegakernelTalkerBackend:
    def __init__(self, tts_model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base"):
        # Load megakernel with TTS talker decoder weights
        self.decoder = MegakernelDecoder(model_name=tts_model_name)
        # Load codec/vocoder from qwen-tts (separate from decoder)
        self.codec = ...  # Extract from Qwen3TTSModel
        
    def text_to_codec_tokens(self, text, language="English", speaker=None):
        # Build prompt: text + codec control tokens
        prompt_ids = self._build_tts_prompt(text, language, speaker)
        # Run megakernel: prompt → codec token stream
        codec_tokens = list(self.decoder.generate_token_ids(prompt_ids, ...))
        return codec_tokens
```

### 4. Wire Codec Tokens → Audio (Codec/Vocoder)

Extract the codec/vocoder from `Qwen3TTSModel` and use it separately:

```python
# After getting codec tokens from megakernel:
audio = self.codec.decode(codec_tokens)  # codec tokens → waveform
```

### 5. Make It Streaming

Instead of generating all codec tokens then decoding:
- Generate codec tokens **streaming** (megakernel already supports this)
- Decode codec tokens → audio **incrementally** (as tokens arrive)
- Yield audio chunks to Pipecat

---

## Implementation Tasks

### Task 1: Extract Codec/Vocoder from Qwen3TTSModel

**Goal:** Get access to the codec/vocoder component separately, so we can feed it codec tokens from the megakernel.

**Approach:**
- Inspect `Qwen3TTSModel` structure: find where `speech_tokenizer` is used
- Create a wrapper that exposes: `decode_codec_tokens(codec_tokens) → audio`
- Test: generate codec tokens manually, decode them, verify audio quality

### Task 2: Build TTS Prompt for Megakernel

**Goal:** Convert text + language + speaker → token ID prompt that the megakernel can decode.

**Approach:**
- Replicate qwen-tts's prompt construction logic (from `generate()`)
- But output token IDs instead of embeddings
- Handle: text tokens, codec control tokens (language, speaker, BOS/EOS), special tokens

### Task 3: Create `MegakernelTalkerBackend` (Replaces Current `Qwen3TTSTalkerBackend`)

**Goal:** New backend that uses megakernel for decoder, qwen-tts codec for audio.

**Interface:**
```python
class MegakernelTalkerBackend:
    def text_to_speech_blocks(self, text, language="English", ref_audio=None, ref_text=None):
        # 1. Build prompt token IDs
        prompt_ids = self._build_prompt(text, language, ...)
        # 2. Stream codec tokens from megakernel
        for codec_token in self.decoder.generate_token_ids(prompt_ids):
            # 3. Decode codec token → audio chunk (if possible incrementally)
            audio_chunk = self.codec.decode_token(codec_token)  # or batch
            yield audio_chunk, sample_rate
```

### Task 4: Streaming Codec Decode

**Goal:** Decode codec tokens incrementally (not wait for full sequence).

**Challenge:** Qwen3-TTS codec may need full sequences. Check if it supports streaming decode or if we need to buffer small windows.

### Task 5: Update `Qwen3TTSPipecatService` to Use New Backend

Replace `Qwen3TTSTalkerBackend` with `MegakernelTalkerBackend` in `pipecat_tts_service.py`.

---

## Files to Create/Modify

1. **`megakernel_tts_backend.py`** (new)
   - `MegakernelTalkerBackend`: megakernel decoder + qwen-tts codec
   - `_build_tts_prompt()`: text → token ID prompt
   - `_extract_codec()`: get codec/vocoder from Qwen3TTSModel

2. **`inference_server.py`** (modify)
   - Replace `Qwen3TTSTalkerBackend` with `MegakernelTalkerBackend` (or make it optional/selectable)

3. **`pipecat_tts_service.py`** (modify)
   - Use `MegakernelTalkerBackend` instead of `Qwen3TTSTalkerBackend`

4. **Tests** (new)
   - `test_megakernel_tts_integration.py`: verify megakernel produces codec tokens, codec decodes to audio, streaming works

---

## Next Steps

1. **Inspect qwen-tts codec/vocoder access:** Can we get `speech_tokenizer` separately? How does it decode codec tokens?
2. **Test megakernel with TTS weights:** Load TTS model, generate tokens, verify they're codec tokens (not text)
3. **Build prompt construction:** Replicate qwen-tts prompt logic but output token IDs
4. **Implement `MegakernelTalkerBackend`:** Wire megakernel → codec tokens → codec decode → audio
5. **Add streaming:** Make codec decode incremental if possible

---

## Questions to Answer

1. **Does the megakernel produce codec tokens when loaded with TTS weights?** Or does it still produce text tokens? (Need to test)
2. **Can qwen-tts codec decode incrementally?** Or does it need full sequences?
3. **What's the exact prompt format?** Need to replicate qwen-tts's `generate()` prompt construction but as token IDs.
4. **Do we need the code_predictor?** Or can we just use the main decoder's codec_head output?
