#!/usr/bin/env python3
"""
Test MegakernelTalkerBackend: megakernel → codec tokens → codec/vocoder → audio.

Verifies streaming (TTFC = time to first chunk) and RTF.
Run on a machine with GPU + qwen-tts + megakernel built:
  python test_megakernel_tts_backend.py
  python test_megakernel_tts_backend.py --wav out.wav
  python test_megakernel_tts_backend.py --codec-chunk-frames 6   # smaller chunks = lower TTFC
"""

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Test MegakernelTalkerBackend (megakernel as talker → audio).")
    parser.add_argument("--wav", type=str, default=None, help="If set, write all chunks to this WAV file.")
    parser.add_argument("--text", type=str, default="Hello.", help="Text to synthesize.")
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--codec-chunk-frames", type=int, default=12, help="Codec frames per decode (default 12 ≈ 1 s). Smaller = lower TTFC.")
    args = parser.parse_args()

    print("Loading MegakernelTalkerBackend (megakernel + TTS codec)...")
    try:
        from megakernel_tts_backend import MegakernelTalkerBackend
    except Exception as e:
        print(f"Import failed: {e}")
        sys.exit(1)

    backend = MegakernelTalkerBackend(verbose=True)
    t_start = time.perf_counter()
    ttfc_ms = None
    chunks = []
    for chunk_tensor, sr in backend.text_to_speech_blocks(
        text=args.text,
        language=args.language,
        max_new_tokens=1024,
        codec_chunk_frames=args.codec_chunk_frames,
    ):
        if ttfc_ms is None:
            ttfc_ms = (time.perf_counter() - t_start) * 1000
        chunks.append((chunk_tensor, sr))
    elapsed = time.perf_counter() - t_start
    total_samples = sum(c.shape[0] for c, _ in chunks)
    sr = chunks[0][1] if chunks else 0
    duration_s = total_samples / sr if sr else 0
    rtf = elapsed / duration_s if duration_s else 0

    print()
    print("Results:")
    print(f"  Chunks: {len(chunks)}, total samples: {total_samples}, sr: {sr}")
    print(f"  TTFC (time to first chunk): {ttfc_ms:.0f} ms" if ttfc_ms is not None else "  TTFC: N/A (no chunks)")
    print(f"  Elapsed: {elapsed:.2f} s, audio duration: {duration_s:.2f} s, RTF: {rtf:.2f}")
    if chunks:
        print("  First chunk shape:", chunks[0][0].shape)
    if ttfc_ms is not None:
        target_ttfc = 90
        print(f"  TTFC < {target_ttfc} ms: {'✓' if ttfc_ms < target_ttfc else '✗'}")

    if args.wav and chunks:
        import numpy as np
        try:
            import soundfile as sf
            parts = []
            for chunk_tensor, _ in chunks:
                if hasattr(chunk_tensor, "numpy"):
                    parts.append(chunk_tensor.numpy())
                else:
                    parts.append(np.asarray(chunk_tensor))
            wav = np.concatenate(parts)
            sf.write(args.wav, wav, sr)
            print(f"  Wrote: {args.wav}")
        except ImportError:
            print("  Install soundfile to write WAV: pip install soundfile")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
