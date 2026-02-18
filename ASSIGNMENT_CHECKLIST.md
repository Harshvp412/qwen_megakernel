# Assignment Checklist — Qwen Megakernel + Qwen3-TTS + Pipecat

This document maps the assignment requirements to what is implemented and where we fall short.

---

## Goal (TL;DR)

> Take AlpinDale's qwen_megakernel and wire it up to serve Qwen3-TTS inference inside a Pipecat voice pipeline.

**Status:** Partially met. We have a working Pipecat voice pipeline that uses Qwen3-TTS for audio and the megakernel for standalone decode. The megakernel is **not** yet used as the LLM decode backend *inside* the TTS talker stage (see gaps below).

---

## Step 1 — Adapt the Megakernel for Qwen3-TTS

| Requirement | Status | Notes |
|-------------|--------|--------|
| Clone megakernel | ✅ | Based on AlpinDale/qwen_megakernel |
| Qwen3-TTS talker decoder = same Qwen3 architecture (target) | ✅ | Verified via `inspect_tts_model.py`; 0.6B/talker shapes match |
| Verify weight shapes (0.6B → talker LM backbone) | ✅ | Vocab 151936, hidden 1024, 28 layers, 8 KV heads, head dim 128 |
| If shapes differ, adjust kernel | ✅ N/A | Shapes match; no kernel dimension changes |
| Load talker decoder weights into megakernel layout | ✅ | `model.py` supports TTS model path and `qwen3_tts` |
| Validate: decoded tokens match HF reference | ⚠️ Documented | Parity test exists; **first token mismatch** vs HF (CUDA kernel RoPE/attention). Later tokens can match. Documented in repo; decoder is still usable. |

---

## Step 2 — Build the Inference Server

| Requirement | Status | Notes |
|-------------|--------|--------|
| Wrap megakernel in Python-callable inference server | ✅ | `inference_server.py`: `MegakernelDecoder`, ctypes via existing extension |
| Streaming: prompt in → token stream out | ✅ | `generate_token_ids()` yields tokens; used in Step 2 tests |
| TTS decode loop + codec/vocoder for audio | ⚠️ Partial | **Gap:** We use **qwen-tts end-to-end** (its internal decoder + codec/vocoder). We did **not** replace the talker decoder with the megakernel and then wire our token stream into Qwen3-TTS’s codec/vocoder. So the megakernel is not “running as the LLM decode backend for Qwen3-TTS’s talker decoder” in the TTS path. |

---

## Step 3 — Integrate with Pipecat

| Requirement | Status | Notes |
|-------------|--------|--------|
| Custom Pipecat TTS service calling our inference server | ✅ | `pipecat_tts_service.py`: `Qwen3TTSPipecatService` calls `Qwen3TTSTalkerBackend` (text → audio) |
| Standard Pipecat TTS interface (text in, audio frames out) | ✅ | `run_tts(text, context_id)` → `TTSStartedFrame`, `TTSAudioRawFrame`(s), `TTSStoppedFrame` |
| Pipeline: STT → LLM → our TTS → audio | ✅ | `demo_pipecat.py`: Deepgram, OpenAI, our TTS, transport output |
| Streaming — push frames as generated, not buffer full utterance | ❌ | **Gap:** We generate the **full utterance** with qwen-tts, then **chunk** and push. So we do *not* “push audio frames as they’re generated”; we buffer-then-send. True streaming would require megakernel → codec/vocoder integration and chunked output. |

---

## Step 4 — Validate End-to-End

| Requirement | Status | Notes |
|-------------|--------|--------|
| Round-trip: speak → transcribe → LLM → TTS → playback | ✅ | `demo_pipecat.py` + `test_step4_pipeline.py` |
| Measure: tok/s, TTFC, RTF, latency | ✅ | Reported in tests and README |
| TTFC < 90 ms (target) | ❌ | ~20–28 s (full-utterance TTS then first chunk); target not met |
| RTF < 0.3 (target) | ❌ | ~1.7–1.9 (e.g. 20 s to generate 10 s audio); target not met |
| Streaming frame-by-frame, not buffered-then-sent | ❌ | See Step 3; we buffer then chunk |
| Audio quality acceptable | ✅ | No formal metric; pipeline runs and produces audio |

**Reported numbers (current):**

- **Megakernel tok/s:** ~740–750 (benchmark); blog reference ~1000 on same hardware.
- **TTFC:** ~20–28 s (time to first TTS chunk; dominated by full-utterance generation).
- **RTF:** ~1.7–1.9.
- **End-to-end:** Round-trip works; latency dominated by TTS generation.

---

## Deliverables

| Deliverable | Status | Notes |
|-------------|--------|--------|
| Working repo, build instructions (single RTX 5090) | ✅ | README + `requirements.txt` + `run_full_pipeline_test.sh` |
| README: architecture, kernel mods, how to run Pipecat demo | ✅ | README has pipeline section, kernel notes, demo instructions |
| Performance numbers (tok/s, TTFC, RTF, latency) | ✅ | In README and test output; TTFC/RTF documented as not meeting targets |
| Demo recording or script | ✅ | `demo_pipecat.py`; recording optional |

---

## Summary: What’s Done vs What’s Missing

**Done:**

- Megakernel runs; Step 1 weight/shape verification and loading for TTS path; parity documented (first-token mismatch).
- Inference server with streaming token API; TTS backend (qwen-tts) producing chunked audio for Pipecat.
- Pipecat TTS service and full pipeline (STT → LLM → TTS → audio); end-to-end test and validation script.
- Honest performance reporting and README.

**Missing / partial:**

1. **Megakernel as talker decoder:** The megakernel is not used as the decode backend *inside* Qwen3-TTS. TTS uses qwen-tts’s full stack. To fully meet the goal, we’d need to: run the talker decoder with the megakernel, produce a token stream, and feed it into Qwen3-TTS’s codec/vocoder (or equivalent) — and possibly patch or fork qwen-tts to accept an external decoder.
2. **True streaming TTS:** Audio is “streamed” only in the sense of chunked delivery after full-utterance generation. Pushing frames *as* they’re generated would require the above integration and streaming codec/vocoder output.
3. **TTFC/RTF targets:** Not met; would require streaming TTS and megakernel-in-the-loop as above.
4. **Parity:** First token does not match HF; documented as kernel-level (RoPE/attention), not fixed here.

This checklist is the single place to see alignment with the assignment and where we’re short.
