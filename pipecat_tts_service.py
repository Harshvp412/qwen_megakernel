#!/usr/bin/env python3
"""
Pipecat TTS service adapter for Qwen3-TTS using the megakernel inference server.

Streams audio from Qwen3TTSTalkerBackend (inference_server) into Pipecat frames.
Requires: pipecat-ai, qwen-tts (optional for import; required for synthesis).
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Optional

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService


def _float32_pcm_to_int16_bytes(audio_tensor) -> bytes:
    """Convert float32 [-1, 1] tensor to 16-bit PCM bytes (little-endian)."""
    import numpy as np

    if hasattr(audio_tensor, "numpy"):
        arr = audio_tensor.numpy()
    else:
        arr = audio_tensor
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767).astype(np.int16)
    return pcm.tobytes()


class Qwen3TTSPipecatService(TTSService):
    """
    Pipecat TTSService that uses Qwen3-TTS via inference_server.Qwen3TTSTalkerBackend.

    Yields TTSStartedFrame, TTSAudioRawFrame(s), TTSStoppedFrame. Audio is streamed
    in chunks from the backend (chunked after full generation; native streaming TBD).
    """

    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        backend=None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._backend = backend

    def _get_backend(self):
        if self._backend is not None:
            return self._backend
        try:
            from inference_server import Qwen3TTSTalkerBackend
            return Qwen3TTSTalkerBackend()
        except Exception as e:
            raise RuntimeError(
                "Qwen3-TTS backend not available. Install qwen-tts and ensure "
                "inference_server.Qwen3TTSTalkerBackend can be instantiated."
            ) from e

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Run TTS for text; yield TTSStartedFrame, TTSAudioRawFrames, TTSStoppedFrame."""
        backend = self._get_backend()
        loop = asyncio.get_event_loop()

        def _generate_chunks():
            return list(backend.text_to_speech_blocks(text))

        try:
            chunks_list = await loop.run_in_executor(None, _generate_chunks)
        except Exception as e:
            raise RuntimeError(f"Qwen3-TTS synthesis failed: {e}") from e

        yield TTSStartedFrame(context_id=context_id)

        for chunk_tensor, sr in chunks_list:
            pcm_bytes = _float32_pcm_to_int16_bytes(chunk_tensor)
            if not pcm_bytes:
                continue
            yield TTSAudioRawFrame(
                audio=pcm_bytes,
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=context_id,
            )

        yield TTSStoppedFrame(context_id=context_id)
