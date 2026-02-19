#!/usr/bin/env python3
"""
Step 1 Test: Validate TTS decoder parity.

Tests:
1. TTS model loads successfully
2. Megakernel decoder can load TTS model (or fallback to Qwen3-0.6B)
3. Basic token generation works
"""

import sys
from pathlib import Path

# Ensure repo root is on path when run as tests/test_step1_tts_parity.py
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def test_tts_model_load():
    """Test 1.1: Can we load the TTS model?"""
    print("=" * 60)
    print("Test 1.1: TTS Model Loading")
    print("=" * 60)
    
    try:
        import qwen_tts
        from transformers import AutoConfig
        
        print("PASS qwen-tts package installed")
        
        # Try to load config
        try:
            config = AutoConfig.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                trust_remote_code=True
            )
            print(f"PASS TTS config loaded: {config.model_type}")
            print(f"  vocab_size: {config.vocab_size}")
            print(f"  hidden_size: {config.hidden_size}")
            print(f"  num_layers: {config.num_hidden_layers}")
            return True
        except Exception as e:
            print(f"FAIL TTS config load failed: {e}")
            print("  → Will use Qwen3-0.6B as fallback")
            return False
            
    except ImportError:
        print("FAIL qwen-tts not installed")
        print("  → Will use Qwen3-0.6B as fallback")
        return False


def test_megakernel_decoder():
    """Test 1.2: Can megakernel decoder load?"""
    print("\n" + "=" * 60)
    print("Test 1.2: Megakernel Decoder Loading")
    print("=" * 60)
    
    try:
        from qwen_megakernel.model import Decoder
        
        # Try TTS model first, fallback to base
        tts_ok = test_tts_model_load()
        model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base" if tts_ok else "Qwen/Qwen3-0.6B"
        
        print(f"\nLoading decoder with model: {model_name}")
        decoder = Decoder(model_name=model_name, verbose=True)
        print("PASS Decoder loaded successfully")
        
        return decoder
        
    except Exception as e:
        print(f"FAIL Decoder load failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_basic_generation(decoder):
    """Test 1.3: Basic token generation"""
    print("\n" + "=" * 60)
    print("Test 1.3: Basic Token Generation")
    print("=" * 60)
    
    if decoder is None:
        print("FAIL Skipping - decoder not loaded")
        return False
    
    prompt = "The capital of France is"
    print(f"Prompt: {repr(prompt)}")
    
    try:
        decoder.reset()
        prompt_ids = decoder.tokenizer.encode(prompt, add_special_tokens=True)
        print(f"Prompt IDs: {prompt_ids}")
        
        # Prefill
        for tid in prompt_ids[:-1]:
            decoder.step(tid)
        
        # Generate first 5 tokens
        generated = []
        cur = prompt_ids[-1]
        for i in range(5):
            cur = decoder.step(cur)
            generated.append(cur)
            text = decoder.tokenizer.decode([cur], skip_special_tokens=True)
            print(f"  Token {i+1}: {cur} → {repr(text)}")
        
        full_text = decoder.tokenizer.decode(generated, skip_special_tokens=True)
        print(f"\nGenerated text: {repr(full_text)}")
        print("PASS Basic generation works")
        return True
        
    except Exception as e:
        print(f"FAIL Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Step 1 tests."""
    print("\n" + "=" * 60)
    print("STEP 1: TTS Model Compatibility & Parity Tests")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1.1: TTS model load
    tts_ok = test_tts_model_load()
    results.append(("TTS Model Load", tts_ok))
    
    # Test 1.2: Megakernel decoder
    decoder = test_megakernel_decoder()
    results.append(("Megakernel Decoder", decoder is not None))
    
    # Test 1.3: Basic generation
    if decoder:
        gen_ok = test_basic_generation(decoder)
        results.append(("Basic Generation", gen_ok))
    else:
        results.append(("Basic Generation", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("STEP 1 TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:30} {status}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("=" * 60))
    if all_passed:
        print("✅ STEP 1 COMPLETE - Ready for Step 2")
    else:
        print("STEP 1 INCOMPLETE - Some tests failed")
        print("   Proceeding anyway with fallback (Qwen3-0.6B)")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
