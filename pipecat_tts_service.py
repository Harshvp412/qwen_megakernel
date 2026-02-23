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
    in chunks from the backend as they're decoded (no full-utterance buffering).
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
        # Base TTSService sets _sample_rate only in start(); when used standalone
        # (e.g. in tests) we need a valid rate for TTSAudioRawFrame.
        self._sample_rate = sample_rate

    def _get_backend(self):
        if self._backend is not None:
            return self._backend
        # Use legacy (HF) TTS when USE_LEGACY_TTS=1 for reliable demo when megakernel TTS has CUDA issues.
        import os
        if os.environ.get("USE_LEGACY_TTS", "").strip().lower() in ("1", "true", "yes"):
            try:
                from inference_server import Qwen3TTSTalkerBackend
                return Qwen3TTSTalkerBackend()
            except Exception as e:
                raise RuntimeError(
                    "USE_LEGACY_TTS=1 but Qwen3TTSTalkerBackend failed. Install qwen-tts."
                ) from e
        # Prefer megakernel-as-talker backend (full compliance: megakernel → codec → audio).
        try:
            from inference_server import get_megakernel_tts_backend
            return get_megakernel_tts_backend()()
        except Exception:
            pass
        try:
            from inference_server import Qwen3TTSTalkerBackend
            return Qwen3TTSTalkerBackend()
        except Exception as e:
            raise RuntimeError(
                "Qwen3-TTS backend not available. Install qwen-tts and ensure "
                "megakernel_tts_backend.MegakernelTalkerBackend or "
                "inference_server.Qwen3TTSTalkerBackend can be instantiated."
            ) from e

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Run TTS for text; yield TTSStartedFrame, TTSAudioRawFrames, TTSStoppedFrame.
        
        Streams audio chunks as they're decoded (no full-utterance buffering).
        Uses a thread + queue to stream from the synchronous backend generator.
        """
        backend = self._get_backend()
        loop = asyncio.get_event_loop()

        yield TTSStartedFrame(context_id=context_id)

        # Stream chunks as they're generated (no buffering)
        # Use thread + queue to bridge sync generator → async generator
        import queue
        import threading

        q = queue.Queue()
        error_holder = [None]

        def _generate():
            """Run backend generator in thread, put chunks in queue."""
            try:
                for chunk_tensor, sr in backend.text_to_speech_blocks(text):
                    q.put((chunk_tensor, sr))
                q.put(None)  # Sentinel to signal completion
            except Exception as e:
                error_holder[0] = e
                q.put(None)  # Signal completion even on error

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

        # Stream chunks from queue as they arrive
        while True:
            # Get chunk from queue (blocks until available)
            item = await loop.run_in_executor(None, q.get)

            if item is None:
                # Generator finished
                if error_holder[0]:
                    raise RuntimeError(
                        f"Qwen3-TTS synthesis failed: {error_holder[0]}"
                    ) from error_holder[0]
                break

            chunk_tensor, sr = item
            pcm_bytes = _float32_pcm_to_int16_bytes(chunk_tensor)
            if pcm_bytes:
                yield TTSAudioRawFrame(
                    audio=pcm_bytes,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )

        yield TTSStoppedFrame(context_id=context_id)
