#!/usr/bin/env python3
"""
Pipecat voice pipeline demo: STT → LLM → Qwen3-TTS (megakernel-backed).

Uses Deepgram (STT), OpenAI (LLM), and our Qwen3TTSPipecatService (TTS).
Requires: pipecat-ai, qwen-tts, and API keys for Deepgram and OpenAI.

Run (use -t webrtc for local URL; open in browser to use laptop mic and speaker):
  export DEEPGRAM_API_KEY=... OPENAI_API_KEY=...
  python demo_pipecat.py -t webrtc

Optional:
  RECORD_DEMO=1         Save conversation audio to recordings/demo_YYYYMMDD_HHMMSS.wav
  USE_LEGACY_TTS=1     Use HF-based TTS (no megakernel) for reliable demo when MK TTS has CUDA issues
  USE_FREE_MODELS=1    Use free local models: Whisper (STT) + Ollama (LLM). No API keys needed.
                       Requires: pip install "pipecat-ai[whisper]" "pipecat-ai[ollama]", and run ollama serve + ollama pull llama2

Or with .env:
  pip install python-dotenv
  python demo_pipecat.py
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from loguru import logger

logger.info("Loading pipeline components...")
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from pipecat_tts_service import Qwen3TTSPipecatService

logger.info("Loading Silero VAD...")
from pipecat.audio.vad.silero import SileroVADAnalyzer
logger.info("Components loaded")

# Optional: use free local models (Whisper STT + Ollama LLM)
USE_FREE_MODELS = os.environ.get("USE_FREE_MODELS", "").strip().lower() in ("1", "true", "yes")
# Auto-enable free models if no API keys (so demo can run without keys)
if not USE_FREE_MODELS and (not os.getenv("DEEPGRAM_API_KEY") or not os.getenv("OPENAI_API_KEY")):
    USE_FREE_MODELS = True
    logger.info("No API keys set; using free local models (Whisper + Ollama)")

# Optional recording
RECORD_DEMO = os.environ.get("RECORD_DEMO", "").strip().lower() in ("1", "true", "yes")


def _create_stt():
    """Create STT: Whisper (free) or Deepgram (API key)."""
    if USE_FREE_MODELS:
        try:
            from pipecat.services.whisper import WhisperSTTService
            return WhisperSTTService(model="base", device="cuda" if __import__("torch").cuda.is_available() else "cpu")
        except ImportError:
            raise RuntimeError(
                "USE_FREE_MODELS=1 requires: pip install \"pipecat-ai[whisper]\""
            ) from None
    from pipecat.services.deepgram.stt import DeepgramSTTService
    key = os.getenv("DEEPGRAM_API_KEY")
    if not key:
        raise RuntimeError("Set DEEPGRAM_API_KEY or use USE_FREE_MODELS=1 (Whisper)")
    return DeepgramSTTService(api_key=key)


def _create_llm():
    """Create LLM: Ollama (free) or OpenAI (API key)."""
    if USE_FREE_MODELS:
        try:
            from pipecat.services.ollama import OLLamaLLMService
            return OLLamaLLMService(model=os.getenv("OLLAMA_MODEL", "llama2"))
        except ImportError:
            raise RuntimeError(
                "USE_FREE_MODELS=1 requires: pip install \"pipecat-ai[ollama]\""
            ) from None
    from pipecat.services.openai.llm import OpenAILLMService
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY or use USE_FREE_MODELS=1 (Ollama)")
    return OpenAILLMService(api_key=key)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    tts_backend = "legacy (HF)" if os.environ.get("USE_LEGACY_TTS", "").strip().lower() in ("1", "true", "yes") else "megakernel-backed"
    mode = "Whisper+Ollama (free)" if USE_FREE_MODELS else "Deepgram+OpenAI"
    logger.info("Starting bot (STT+LLM=%s, TTS=Qwen3-TTS [%s])", mode, tts_backend)

    stt = _create_stt()
    llm = _create_llm()

    try:
        tts = Qwen3TTSPipecatService(sample_rate=24000)
    except RuntimeError as e:
        logger.error(
            "Qwen3-TTS not available. Install qwen-tts and ensure GPU/model is set up. "
            f"Error: {e}"
        )
        raise

    messages = [
        {
            "role": "system",
            "content": "You are a friendly AI assistant. Respond naturally and keep answers short and conversational.",
        },
    ]

    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # Optional: record conversation to WAV for demo recording
    record_chunks = []  # list of (audio_bytes, sample_rate, num_channels)
    audiobuffer = None
    if RECORD_DEMO:
        try:
            from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
            audiobuffer = AudioBufferProcessor(
                num_channels=1,
                enable_turn_audio=False,
                user_continuous_stream=True,
            )
            logger.info("Recording enabled: conversation will be saved to recordings/")
        except Exception as e:
            logger.warning("RECORD_DEMO=1 but AudioBufferProcessor failed: %s", e)
            audiobuffer = None

    pipeline_stages = [
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
    ]
    if audiobuffer is not None:
        pipeline_stages.append(audiobuffer)
    pipeline_stages.append(assistant_aggregator)
    pipeline = Pipeline(pipeline_stages)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_out_sample_rate=24000,
        ),
    )

    if audiobuffer is not None:
        @audiobuffer.event_handler("on_audio_data")
        async def on_audio_data(buffer, audio, sample_rate, num_channels):
            record_chunks.append((bytes(audio), sample_rate, num_channels))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        if audiobuffer is not None:
            await audiobuffer.start_recording()
        messages.append({"role": "system", "content": "Say hello and briefly introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        if RECORD_DEMO and record_chunks:
            import wave
            from datetime import datetime
            recordings_dir = Path(__file__).resolve().parent / "recordings"
            recordings_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = recordings_dir / f"demo_{timestamp}.wav"
            sr = record_chunks[0][1]
            nc = record_chunks[0][2]
            with wave.open(str(out_path), "wb") as wf:
                wf.setnchannels(nc)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                for chunk_bytes, _, _ in record_chunks:
                    wf.writeframes(chunk_bytes)
            logger.info("Saved demo recording to %s", out_path)
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


def _get_daily_params():
    """Lazy import so Daily SDK is only required when using -t daily."""
    from pipecat.transports.daily.transport import DailyParams
    return DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    )


async def bot(runner_args: RunnerArguments):
    transport_params = {
        "daily": _get_daily_params,
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
