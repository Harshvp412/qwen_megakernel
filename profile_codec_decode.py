#!/usr/bin/env python3
"""
Profile codec decode latency vs number of frames.
Measures speech_tokenizer.decode() for 1, 2, 4, 8, 12 frames to find
the minimum decode time and whether we can get TTFC < 90 ms with a small first chunk.

Usage (GPU + qwen-tts):
  python profile_codec_decode.py
  python profile_codec_decode.py --repeats 5
"""

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Profile codec decode latency vs frame count.")
    parser.add_argument("--repeats", type=int, default=3, help="Decode repeats per frame count (for averaging).")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base", help="TTS model for speech_tokenizer.")
    args = parser.parse_args()

    print("Loading TTS model (speech_tokenizer)...")
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as e:
        print(f"Import error: {e}")
        sys.exit(1)

    tts = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    speech_tokenizer = tts.model.speech_tokenizer

    num_layers = 16
    codebook_size = 2048
    try:
        dec = getattr(speech_tokenizer, "model", None) and getattr(speech_tokenizer.model, "decoder", None)
        if dec is not None and hasattr(dec, "config"):
            num_layers = getattr(dec.config, "num_quantizers", num_layers)
            codebook_size = getattr(dec.config, "codebook_size", codebook_size)
    except Exception:
        pass
    print(f"Codec: num_layers={num_layers}, codebook_size={codebook_size}\n")

    import numpy as np
    frame_counts = [1, 2, 4, 6, 8, 12]
    target_ttfc_ms = 90

    # Warmup
    dummy = np.zeros((2, num_layers), dtype=np.int64)
    try:
        speech_tokenizer.decode([{"audio_codes": dummy}])
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print("Decode latency (ms) vs frame count:")
    print("-" * 50)
    for n in frame_counts:
        codes = np.zeros((n, num_layers), dtype=np.int64)
        times_ms = []
        for _ in range(args.repeats):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            speech_tokenizer.decode([{"audio_codes": codes}])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)
        avg = sum(times_ms) / len(times_ms)
        ok = "✓" if avg < target_ttfc_ms else "✗"
        print(f"  {n:3d} frames  →  {avg:7.1f} ms avg  {ok} (target < {target_ttfc_ms} ms)")
    print("-" * 50)
    print("\nRecommendation: Use first_chunk_frames = smallest n where latency < 90 ms for TTFC.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
