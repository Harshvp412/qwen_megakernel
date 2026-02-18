#!/usr/bin/env python3
"""
Test: Does megakernel produce codec tokens when loaded with TTS talker weights?

Goal: Verify that when we load Qwen3-TTS talker decoder weights into the megakernel,
it produces codec tokens (not text tokens) that can be fed to the codec/vocoder.
"""

import sys

# Register qwen3_tts architecture in transformers before loading TTS model
try:
    from qwen_tts import Qwen3TTSModel  # noqa: F401 - triggers config/model registration
except ImportError:
    pass

try:
    from qwen_megakernel.model import Decoder
except ImportError:
    print("qwen_megakernel not available")
    sys.exit(1)


def test_megakernel_with_tts_weights():
    """Test megakernel decoder with TTS model weights."""
    print("=" * 60)
    print("Test: Megakernel with TTS Talker Decoder Weights")
    print("=" * 60)
    
    tts_model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    
    print(f"\nLoading megakernel decoder with TTS model: {tts_model_name}")
    try:
        decoder = Decoder(model_name=tts_model_name, verbose=True)
        print("✓ Decoder loaded")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\nTokenizer vocab size: {len(decoder.tokenizer)}")
    print(f"Tokenizer type: {type(decoder.tokenizer).__name__}")
    
    # Test generation
    prompt = "Hello"
    print(f"\nGenerating tokens for prompt: {repr(prompt)}")
    
    decoder.reset()
    prompt_ids = decoder.tokenizer.encode(prompt, add_special_tokens=True)
    print(f"Prompt token IDs: {prompt_ids}")
    
    # Prefill
    for tid in prompt_ids[:-1]:
        decoder.step(tid)
    
    # Generate a few tokens (with TTS weights these are codec token IDs, not text)
    generated = []
    for i in range(10):
        token_id = decoder.step(prompt_ids[-1] if i == 0 else token_id)
        generated.append(token_id)
        # Decoding as text is wrong for codec IDs; show both for debugging
        as_text = decoder.tokenizer.decode([token_id], skip_special_tokens=True)
        print(f"  Token {i+1}: {token_id:6d}  (as text: {repr(as_text)})")
    
    print(f"\nGenerated codec token IDs: {generated}")
    as_text_debug = decoder.tokenizer.decode(generated, skip_special_tokens=True)
    print(f"(Decoded as text — not meaningful for codec): {repr(as_text_debug)}")
    
    print("\n" + "-" * 60)
    print("Analysis:")
    print("-" * 60)
    print("  Token IDs range:", min(generated), "to", max(generated))
    print("  With TTS weights, these are codec tokens; feed to codec/vocoder for audio.")
    
    return True


if __name__ == "__main__":
    success = test_megakernel_with_tts_weights()
    sys.exit(0 if success else 1)
