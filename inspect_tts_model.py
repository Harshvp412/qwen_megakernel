#!/usr/bin/env python3
"""Inspect Qwen3-TTS talker decoder architecture and compare with Qwen3-0.6B."""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def inspect_model(model_name: str):
    """Inspect model architecture and print key dimensions."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {model_name}")
    print(f"{'='*60}\n")
    
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"ERROR loading config: {e}")
        print("Try: pip install qwen-tts")
        return None
    
    print("Architecture:")
    print(f"  model_type: {getattr(config, 'model_type', 'unknown')}")
    print(f"  vocab_size: {getattr(config, 'vocab_size', 'N/A')}")
    print(f"  hidden_size: {getattr(config, 'hidden_size', 'N/A')}")
    print(f"  num_hidden_layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
    print(f"  num_attention_heads: {getattr(config, 'num_attention_heads', 'N/A')}")
    print(f"  num_key_value_heads: {getattr(config, 'num_key_value_heads', 'N/A')}")
    print(f"  intermediate_size: {getattr(config, 'intermediate_size', 'N/A')}")
    print(f"  max_position_embeddings: {getattr(config, 'max_position_embeddings', 'N/A')}")
    print(f"  rope_theta: {getattr(config, 'rope_theta', 'N/A')}")
    
    # Check for talker decoder specifically
    if hasattr(config, 'decoder_config'):
        print("\nDecoder config found:")
        dec_config = config.decoder_config
        print(f"  decoder vocab_size: {getattr(dec_config, 'vocab_size', 'N/A')}")
        print(f"  decoder hidden_size: {getattr(dec_config, 'hidden_size', 'N/A')}")
        print(f"  decoder num_layers: {getattr(dec_config, 'num_hidden_layers', 'N/A')}")
    
    # Try to load state dict to check actual weight shapes
    try:
        print("\nLoading model to inspect weights...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # Load on CPU to check shapes
            trust_remote_code=True
        )
        
        state = model.state_dict()
        print("\nKey weight shapes:")
        
        # Check embedding
        if "model.embed_tokens.weight" in state:
            embed_shape = state["model.embed_tokens.weight"].shape
            print(f"  embed_tokens.weight: {embed_shape}")
        
        # Check LM head
        if "lm_head.weight" in state:
            lm_head_shape = state["lm_head.weight"].shape
            print(f"  lm_head.weight: {lm_head_shape}")
        elif "model.embed_tokens.weight" in state:
            print(f"  lm_head.weight: (tied to embed_tokens)")
        
        # Check first layer
        if "model.layers.0.input_layernorm.weight" in state:
            print(f"  layers.0.input_layernorm.weight: {state['model.layers.0.input_layernorm.weight'].shape}")
        if "model.layers.0.self_attn.q_proj.weight" in state:
            q_shape = state["model.layers.0.self_attn.q_proj.weight"].shape
            print(f"  layers.0.self_attn.q_proj.weight: {q_shape}")
        if "model.layers.0.self_attn.k_proj.weight" in state:
            k_shape = state["model.layers.0.self_attn.k_proj.weight"].shape
            print(f"  layers.0.self_attn.k_proj.weight: {k_shape}")
        
        # Count layers
        layer_keys = [k for k in state.keys() if k.startswith("model.layers.")]
        if layer_keys:
            layer_nums = set()
            for k in layer_keys:
                parts = k.split(".")
                if len(parts) >= 3 and parts[2].isdigit():
                    layer_nums.add(int(parts[2]))
            print(f"  Number of layers (from state_dict): {max(layer_nums) + 1 if layer_nums else 'N/A'}")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return config
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return config

def compare_with_megakernel(config):
    """Compare TTS model config with megakernel constants."""
    print(f"\n{'='*60}")
    print("Comparison with Megakernel Constants")
    print(f"{'='*60}\n")
    
    MK_CONSTANTS = {
        "NUM_LAYERS": 28,
        "NUM_KV_HEADS": 8,
        "HEAD_DIM": 128,
        "HIDDEN_SIZE": 1024,
        "INTERMEDIATE_SIZE": 3072,
        "VOCAB_SIZE": 151936,
    }
    
    model_config = {
        "NUM_LAYERS": getattr(config, "num_hidden_layers", None),
        "NUM_KV_HEADS": getattr(config, "num_key_value_heads", None),
        "HEAD_DIM": getattr(config, "head_dim", None) or (getattr(config, "hidden_size", 0) // getattr(config, "num_attention_heads", 1) if getattr(config, "num_attention_heads", 0) else None),
        "HIDDEN_SIZE": getattr(config, "hidden_size", None),
        "INTERMEDIATE_SIZE": getattr(config, "intermediate_size", None),
        "VOCAB_SIZE": getattr(config, "vocab_size", None),
    }
    
    print("Constant                    Megakernel    TTS Model    Match")
    print("-" * 60)
    all_match = True
    for key in MK_CONSTANTS:
        mk_val = MK_CONSTANTS[key]
        tt_val = model_config[key]
        match = "✓" if mk_val == tt_val else "✗"
        if mk_val != tt_val:
            all_match = False
        print(f"{key:28} {mk_val:12} {str(tt_val):12} {match}")
    
    print()
    if all_match:
        print("✅ All dimensions match! Megakernel can be used as-is.")
    else:
        print("⚠️  Dimensions differ. Kernel constants need adjustment.")
        print("   Check csrc/kernel.cu and qwen_megakernel/model.py")
    
    return all_match

if __name__ == "__main__":
    import sys
    
    # Check Qwen3-0.6B (baseline)
    print("=" * 60)
    print("BASELINE: Qwen3-0.6B")
    print("=" * 60)
    base_config = inspect_model("Qwen/Qwen3-0.6B")
    if base_config:
        compare_with_megakernel(base_config)
    
    # Check Qwen3-TTS models
    # Note: These require qwen-tts package to be installed
    tts_models = [
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ]
    
    for tts_model in tts_models:
        try:
            config = inspect_model(tts_model)
            if config:
                matches = compare_with_megakernel(config)
                if matches:
                    print(f"\n✅ {tts_model} is compatible with megakernel!")
                    break
        except Exception as e:
            print(f"\n⚠️  Could not inspect {tts_model}: {e}")
            continue
