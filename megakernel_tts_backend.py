#!/usr/bin/env python3
"""
Megakernel-as-talker TTS backend: megakernel decode → codec token stream → Qwen3-TTS codec/vocoder → audio.

Full flow: text (+ language/speaker) → prompt token IDs → megakernel (talker weights) → codec tokens
→ speech_tokenizer.decode → audio chunks.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch

# Lazy imports for qwen_tts and MegakernelDecoder to keep CPU-only installs working.


def build_tts_prompt_token_ids(
    talker_config,
    language: str = "English",
    speaker: Optional[str] = None,
    *,
    max_control_tokens: int = 32,
) -> List[int]:
    """
    Build a sequence of codec token IDs used as the talker prompt (control tokens only).
    Uses the same IDs as qwen-tts: codec_think, language, pad, bos, etc.
    Text conditioning would require one forward from qwen-tts to get the first codec token;
    for now we use control tokens only so the megakernel generates from a fixed prefix.
    """
    cfg = talker_config
    lang_lower = (language or "english").lower()
    codec_lang = getattr(cfg, "codec_language_id", None) or {}
    if not isinstance(codec_lang, dict):
        codec_lang = {}
    language_id = codec_lang.get(lang_lower) or codec_lang.get("english") or getattr(cfg, "codec_bos_id", 0)
    # Control sequence: think_bos, language, think_eos, pad, bos (simplified from modeling)
    codec_think_id = getattr(cfg, "codec_think_id", None)
    codec_think_bos_id = getattr(cfg, "codec_think_bos_id", None)
    codec_think_eos_id = getattr(cfg, "codec_think_eos_id", None)
    codec_pad_id = getattr(cfg, "codec_pad_id", None)
    codec_bos_id = getattr(cfg, "codec_bos_id", None)
    if codec_think_id is not None and codec_think_bos_id is not None:
        prompt = [codec_think_id, codec_think_bos_id, language_id, codec_think_eos_id]
    else:
        prompt = []
    if codec_pad_id is not None:
        prompt.append(codec_pad_id)
    if codec_bos_id is not None:
        prompt.append(codec_bos_id)
    # Optional speaker token (single id from spk_id dict)
    if speaker and getattr(cfg, "spk_id", None) and speaker.lower() in cfg.spk_id:
        prompt.append(cfg.spk_id[speaker.lower()])
    return prompt[:max_control_tokens]


class MegakernelTalkerBackend:
    """
    TTS backend that uses the megakernel as the talker decoder:
    prompt (codec control token IDs) → megakernel → codec token stream → codec/vocoder → audio.
    """

    def __init__(
        self,
        tts_model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        verbose: bool = False,
    ) -> None:
        self._tts_model_name = tts_model_name
        self._verbose = verbose
        self._decoder = None
        self._tts_model = None
        self._talker_config = None
        self._speech_tokenizer = None
        self._codec_eos_id = None

    def _ensure_loaded(self) -> None:
        if self._decoder is not None:
            return
        from inference_server import MegakernelDecoder

        self._decoder = MegakernelDecoder(model_name=self._tts_model_name, verbose=self._verbose)
        from qwen_tts import Qwen3TTSModel  # type: ignore[import-untyped]

        tts = Qwen3TTSModel.from_pretrained(
            self._tts_model_name,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self._tts_model = tts
        talker = getattr(tts, "talker", None) or getattr(tts.model, "talker", None)
        if talker is None:
            raise AttributeError("TTS model has no talker (model.model.talker)")
        self._talker_config = talker.config
        self._speech_tokenizer = tts.model.speech_tokenizer
        self._codec_eos_id = getattr(self._talker_config, "codec_eos_token_id", None)

    def build_prompt_ids(
        self,
        text: str,
        language: str = "English",
        speaker: Optional[str] = None,
    ) -> List[int]:
        """Build codec control token ID sequence for the megakernel prompt (no text→codec yet)."""
        self._ensure_loaded()
        return build_tts_prompt_token_ids(
            self._talker_config,
            language=language,
            speaker=speaker,
        )

    def generate_codec_tokens(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 2048,
        stop_on_eos: bool = True,
    ) -> Iterable[int]:
        """Stream codec token IDs from the megakernel given prompt token IDs."""
        self._ensure_loaded()
        dec = self._decoder._decoder
        dec.reset()
        for tid in prompt_ids[:-1]:
            dec.step(tid)
        cur = prompt_ids[-1] if prompt_ids else 0
        for _ in range(max_new_tokens):
            cur = dec.step(cur)
            yield cur
            if stop_on_eos and self._codec_eos_id is not None and cur == self._codec_eos_id:
                break

    def codec_tokens_to_audio(
        self,
        codec_token_ids: List[int],
    ) -> Tuple[torch.Tensor, int]:
        """Decode codec token IDs to waveform using Qwen3-TTS speech_tokenizer."""
        self._ensure_loaded()
        import numpy as np

        # speech_tokenizer.decode expects list of {"audio_codes": c}; c is (T,) or (T, num_codebooks)
        codes = np.array(codec_token_ids, dtype=np.int64)
        if codes.ndim == 1:
            codes = np.expand_dims(codes, axis=1)
        wavs_list, sample_rate = self._speech_tokenizer.decode([{"audio_codes": codes}])
        wav = wavs_list[0]
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).float()
        return wav, sample_rate

    def text_to_speech_blocks(
        self,
        text: str,
        language: str = "English",
        speaker: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        chunk_size_samples: int = 16000,
        max_new_tokens: int = 2048,
    ) -> Iterable[Tuple[torch.Tensor, int]]:
        """
        Generate audio from text using megakernel as talker decoder.
        Yields (audio_chunk_tensor, sample_rate) for streaming.
        """
        self._ensure_loaded()
        prompt_ids = self.build_prompt_ids(text=text, language=language, speaker=speaker)
        if not prompt_ids:
            prompt_ids = [getattr(self._talker_config, "codec_bos_id", 0)]
        codec_ids = list(
            self.generate_codec_tokens(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                stop_on_eos=True,
            )
        )
        wav, sr = self.codec_tokens_to_audio(codec_ids)
        num_samples = wav.shape[0]
        for start in range(0, num_samples, chunk_size_samples):
            end = min(start + chunk_size_samples, num_samples)
            yield wav[start:end], sr
