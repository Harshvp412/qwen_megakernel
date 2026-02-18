#!/usr/bin/env python3
"""
Pipecat voice pipeline demo: STT → LLM → Qwen3-TTS (megakernel-backed).

Uses Deepgram (STT), OpenAI (LLM), and our Qwen3TTSPipecatService (TTS).
Requires: pipecat-ai, qwen-tts, and API keys for Deepgram and OpenAI.

Run:
  export DEEPGRAM_API_KEY=... OPENAI_API_KEY=...
  python demo_pipecat.py

Or with .env:
  pip install python-dotenv
  python demo_pipecat.py
"""

import os

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
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

from pipecat_tts_service import Qwen3TTSPipecatService

logger.info("Loading Silero VAD...")
from pipecat.audio.vad.silero import SileroVADAnalyzer
logger.info("✅ Components loaded")


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot (STT=Deepgram, LLM=OpenAI, TTS=Qwen3-TTS)")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

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

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_out_sample_rate=24000,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        messages.append({"role": "system", "content": "Say hello and briefly introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
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
