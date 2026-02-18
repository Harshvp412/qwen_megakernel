#!/usr/bin/env python3
"""
Streaming inference server utilities for the Qwen megakernel.

Goals (aligned with assignment brief):
  - Provide an efficient, reusable streaming decode API around `qwen_megakernel.model.Decoder`.
  - Make it easy to plug this into a higher-level TTS pipeline (e.g., Qwen3‑TTS + Pipecat).

This module is intentionally pure-Python so it can be:
  - Imported directly in a web server (FastAPI, Flask, etc.).
  - Used inside a Pipecat TTS service adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, List, Optional

import torch

from qwen_megakernel.model import Decoder


@dataclass
class DecodeConfig:
    """Configuration for a single decode request."""

    max_new_tokens: int = 128
    eos_token_id: Optional[int] = None
    stop_on_eos: bool = True


class MegakernelDecoder:
    """
    Thin, efficient wrapper around `qwen_megakernel.model.Decoder`.

    Design goals:
      - Reuse a single Decoder instance per process (weights stay resident in VRAM).
      - Provide a *streaming* token generator interface.
      - Keep CPU ↔ GPU syncs to a minimum (one per step, as imposed by kernel API).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        verbose: bool = False,
    ) -> None:
        # Load weights once; this allocates all GPU buffers.
        self._decoder = Decoder(model_name=model_name, verbose=verbose)
        self._tokenizer = self._decoder.tokenizer

        # Cache a few frequently-used ids.
        self._eos_id = getattr(self._tokenizer, "eos_token_id", None)

    @property
    def tokenizer(self):
        return self._tokenizer

    def _encode_prompt(self, prompt: str) -> List[int]:
        return self._tokenizer.encode(prompt, add_special_tokens=True)

    def generate_token_ids(
        self,
        prompt: str,
        config: Optional[DecodeConfig] = None,
    ) -> Generator[int, None, None]:
        """
        Stream token ids from the megakernel, given a text prompt.

        This mirrors the logic used in `compare_tokens.py` and `parity_reference.py`:
          - Encode prompt → prompt_ids
          - Prefill all but the last prompt id through the kernel (build KV cache)
          - Feed the last prompt id to step() to get the first generated token
          - Continue calling step() autoregressively
        """
        if config is None:
            config = DecodeConfig()

        dec = self._decoder
        tok = self._tokenizer

        prompt_ids = self._encode_prompt(prompt)
        if not prompt_ids:
            return

        # Allow caller to override eos token; otherwise use tokenizer default.
        eos_id = config.eos_token_id
        if eos_id is None:
            eos_id = self._eos_id

        # Reset KV cache and position for this request.
        dec.reset()

        # Prefill all but the last prompt token.
        for tid in prompt_ids[:-1]:
            dec.step(tid)

        # Now step from the last prompt token to generate max_new_tokens.
        cur = prompt_ids[-1]
        for _ in range(config.max_new_tokens):
            cur = dec.step(cur)
            yield cur
            if config.stop_on_eos and eos_id is not None and cur == eos_id:
                break

    def generate_text(
        self,
        prompt: str,
        config: Optional[DecodeConfig] = None,
    ) -> str:
        """Convenience helper: return the decoded string instead of raw ids."""
        ids = list(self.generate_token_ids(prompt, config=config))
        return self._tokenizer.decode(ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# (Optional) TTS integration scaffold
# ---------------------------------------------------------------------------

class Qwen3TTSTalkerBackend:
    """
    Scaffold for integrating the megakernel decoder into a Qwen3‑TTS talker pipeline.

    IMPORTANT:
      - The public `qwen_tts.Qwen3TTSModel` API currently exposes end‑to‑end
        text → audio methods (e.g., `generate`, `generate_voice_clone`) but does
        not yet provide a simple hook to swap out the internal talker decoder.
      - Wiring the megakernel in as the *internal* decoder therefore requires
        deeper integration with `qwen_tts` internals than is appropriate here.

    This class is provided as a clear extension point:
      - It exposes a clean, streaming token interface (`MegakernelDecoder`).
      - A future version can override / patch Qwen3‑TTS internals to call
        `generate_token_ids()` instead of the stock HF decoder.
    """

    def __init__(
        self,
        decoder: Optional[MegakernelDecoder] = None,
        tts_model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ) -> None:
        self._decoder = decoder or MegakernelDecoder(
            model_name="Qwen/Qwen3-0.6B", verbose=False
        )

        # Lazily load qwen-tts so that local CPU installs without TTS still work.
        try:
            from qwen_tts import Qwen3TTSModel  # type: ignore
        except ImportError:
            raise RuntimeError(
                "qwen-tts is not installed. Install with `pip install qwen-tts` "
                "to enable full TTS integration."
            )

        self._tts_model = Qwen3TTSModel.from_pretrained(
            tts_model_name,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    def text_to_speech_blocks(
        self,
        text: str,
        language: str = "English",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        chunk_size_samples: int = 16000,  # ~1 second at 16kHz
    ) -> Iterable[tuple[torch.Tensor, int]]:
        """
        High-level TTS entry point: text → stream of audio blocks (PCM).

        NOTE: Official qwen-tts does not support streaming yet. This method:
          1. Generates full audio using Qwen3-TTS
          2. Chunks it into smaller blocks for streaming compatibility
          3. Yields (audio_chunk, sample_rate) tuples

        The streaming contract matches what Pipecat expects: an iterator
        of audio tensors with sample rate.

        TODO: When qwen-tts adds native streaming, or when we integrate
        the megakernel decoder directly into the TTS pipeline, this can
        yield real-time chunks instead of post-processing.
        """
        # Generate full audio (non-streaming for now)
        if ref_audio is not None and ref_text is not None:
            wavs, sr = self._tts_model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
        else:
            wavs, sr = self._tts_model.generate(
                text=text,
                language=language,
            )

        # Convert to tensor if needed
        if isinstance(wavs, list):
            wav = torch.cat([torch.tensor(w, dtype=torch.float32) for w in wavs], dim=0)
        else:
            wav = torch.tensor(wavs, dtype=torch.float32) if not isinstance(wavs, torch.Tensor) else wavs

        # Chunk into streaming blocks
        num_samples = wav.shape[0]
        for start in range(0, num_samples, chunk_size_samples):
            end = min(start + chunk_size_samples, num_samples)
            chunk = wav[start:end]
            yield chunk, sr


if __name__ == "__main__":
    # Simple manual test (text → tokens) when run as a script.
    import argparse

    parser = argparse.ArgumentParser(description="Megakernel streaming decode demo.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model id for the decoder backbone.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Prompt to decode from.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
        help="Maximum number of new tokens to generate.",
    )
    args = parser.parse_args()

    mk = MegakernelDecoder(model_name=args.model, verbose=True)
    cfg = DecodeConfig(max_new_tokens=args.max_new_tokens)

    ids = list(mk.generate_token_ids(args.prompt, config=cfg))
    text = mk.tokenizer.decode(ids, skip_special_tokens=True)

    print("Prompt:", repr(args.prompt))
    print("Generated token ids:", ids)
    print("Generated text:", repr(text))

