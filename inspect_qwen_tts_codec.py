#!/usr/bin/env python3
"""
Inspect Qwen3-TTS codec/vocoder structure to understand how to use it separately.

Goal: Extract the codec/vocoder component so we can feed it codec tokens from the megakernel.
"""

import sys

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("qwen-tts not installed. Install with: pip install qwen-tts")
    sys.exit(1)

import torch


def inspect_codec_structure():
    """Inspect Qwen3TTSModel to find codec/vocoder access."""
    print("=" * 60)
    print("Inspecting Qwen3-TTS Model Structure")
    print("=" * 60)
    
    model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    print(f"\nLoading model: {model_name}")
    
    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    
    print("\n" + "-" * 60)
    print("Model Components (top-level attributes):")
    print("-" * 60)
    for name in dir(model):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(model, name)
            if not callable(obj):
                print(f"  {name}: {type(obj).__name__}")
        except Exception:
            pass
    
    print("\n" + "-" * 60)
    print("Talker Components:")
    print("-" * 60)
    if hasattr(model, 'talker'):
        talker = model.talker
        if hasattr(talker, 'named_children'):
            for name, module in talker.named_children():
                print(f"  talker.{name}: {type(module).__name__}")
                if name == 'model' and hasattr(module, 'layers'):
                    print(f"    (decoder layers: {len(module.layers)})")
        else:
            for name in dir(talker):
                if name.startswith("_"): continue
                try:
                    obj = getattr(talker, name)
                    if not callable(obj):
                        print(f"  talker.{name}: {type(obj).__name__}")
                except Exception:
                    pass
    
    print("\n" + "-" * 60)
    print("Speech Tokenizer (Codec/Vocoder):")
    print("-" * 60)
    if hasattr(model, 'speech_tokenizer'):
        print(f"  speech_tokenizer: {type(model.speech_tokenizer).__name__}")
        print(f"  Methods: {[m for m in dir(model.speech_tokenizer) if not m.startswith('_')]}")
    else:
        print("  ⚠️  speech_tokenizer not found as direct attribute")
        print("  Checking if it's loaded separately...")
    
    print("\n" + "-" * 60)
    print("Testing generate_voice_clone to understand flow:")
    print("-" * 60)
    print("  (This will show what generate() returns)")
    
    # Try a minimal generation to see the structure
    try:
        text = "Hello"
        ref_text = "Hello, this is a reference."
        # Create minimal ref audio (silence)
        import numpy as np
        ref_audio = np.zeros(24000, dtype=np.float32)  # 1 second at 24kHz
        
        print(f"\nGenerating audio for: {repr(text)}")
        wavs, sr = model.generate_voice_clone(
            text=text,
            language="English",
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        print(f"  Output: wavs type={type(wavs)}, sr={sr}")
        if isinstance(wavs, list):
            print(f"  wavs length: {len(wavs)}")
            print(f"  First element shape: {wavs[0].shape if hasattr(wavs[0], 'shape') else 'N/A'}")
        elif hasattr(wavs, 'shape'):
            print(f"  wavs shape: {wavs.shape}")
    except Exception as e:
        print(f"  ⚠️  Generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-" * 60)
    print("Key Findings:")
    print("-" * 60)
    print("  1. Need to find how to access codec/vocoder separately")
    print("  2. Need to understand codec token format (from talker.generate())")
    print("  3. Need to test if codec can decode incrementally")


if __name__ == "__main__":
    inspect_codec_structure()
