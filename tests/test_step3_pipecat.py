#!/usr/bin/env python3
"""
Step 3 tests: Pipecat TTS service (no full pipeline or API keys required).

- Test 3.1: Qwen3TTSPipecatService.run_tts() yields correct frame sequence and audio.
- Optional: write synthesized audio to a WAV file for manual listening.

Run: python test_step3_pipecat.py [--write-wav out.wav]
"""

import argparse
import asyncio
import sys


async def test_tts_service_run_tts(write_wav: str | None) -> bool:
    """Test Qwen3TTSPipecatService: run_tts yields Started -> AudioRaw* -> Stopped."""
    print("\n" + "=" * 60)
    print("Test 3.1: Qwen3TTSPipecatService.run_tts()")
    print("=" * 60)

    try:
        from pipecat.frames.frames import (
            TTSAudioRawFrame,
            TTSStartedFrame,
            TTSStoppedFrame,
        )
    except ImportError as e:
        print(f"[SKIP]  Skipped: pipecat not installed: {e}")
        print("   Install with: pip install pipecat-ai")
        return None  # skip

    try:
        from pipecat_tts_service import Qwen3TTSPipecatService
    except ImportError as e:
        print(f"[SKIP]  Skipped: pipecat_tts_service not importable: {e}")
        if "websockets" in str(e):
            print("   Install with: pip install websockets")
        return None  # skip

    try:
        tts = Qwen3TTSPipecatService(sample_rate=24000)
    except RuntimeError as e:
        if "not available" in str(e) or "qwen-tts" in str(e).lower():
            print(f"[SKIP]  TTS backend not available: {e}")
            print("   Install qwen-tts to run this test.")
            return None
        raise

    text = "Hello. This is a quick test."
    context_id = "test-context-1"
    frames = []
    try:
        async for f in tts.run_tts(text, context_id):
            frames.append(f)
    except RuntimeError as e:
        if "not available" in str(e) or "qwen-tts" in str(e).lower() or "qwen_tts" in str(e):
            print(f"[SKIP]  TTS backend not available: {e}")
            print("   Install qwen-tts to run this test.")
            return None
        raise

    # Assert frame sequence
    if not frames:
        print("FAIL No frames yielded")
        return False

    if not isinstance(frames[0], TTSStartedFrame):
        print(f"FAIL First frame should be TTSStartedFrame, got {type(frames[0]).__name__}")
        return False
    if not isinstance(frames[-1], TTSStoppedFrame):
        print(f"FAIL Last frame should be TTSStoppedFrame, got {type(frames[-1]).__name__}")
        return False

    audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    if not audio_frames:
        print("FAIL No TTSAudioRawFrame yielded")
        return False

    total_bytes = sum(len(f.audio) for f in audio_frames)
    if total_bytes == 0:
        print("FAIL All audio frames are empty")
        return False

    sr = audio_frames[0].sample_rate
    if not sr:
        print("FAIL TTSAudioRawFrame has sample_rate=0 (service may not have been started in a pipeline)")
        return False
    duration_sec = total_bytes / (2 * sr)  # 16-bit = 2 bytes per sample, mono
    print(f"PASS Frame sequence: TTSStartedFrame → {len(audio_frames)} TTSAudioRawFrame(s) → TTSStoppedFrame")
    print(f"  Total audio: {total_bytes} bytes, {duration_sec:.2f}s @ {sr} Hz")

    if write_wav:
        import wave
        with wave.open(write_wav, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            for f in audio_frames:
                wav.writeframes(f.audio)
        print(f"  Wrote: {write_wav} (play with any media player)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Step 3 Pipecat TTS tests")
    parser.add_argument("--write-wav", type=str, default=None, help="Write output audio to this WAV file")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("STEP 3: Pipecat TTS Service Tests")
    print("=" * 60)

    result = asyncio.run(test_tts_service_run_tts(args.write_wav))

    print("\n" + "=" * 60)
    print("STEP 3 TEST SUMMARY")
    print("=" * 60)
    if result is None:
        print("  Qwen3TTSPipecatService.run_tts()   SKIP (see message above)")
    elif result:
        print("  Qwen3TTSPipecatService.run_tts()   PASS")
    else:
        print("  Qwen3TTSPipecatService.run_tts()   FAIL")
    print("=" * 60 + "\n")

    if result is False:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
