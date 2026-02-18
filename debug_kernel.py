#!/usr/bin/env python3
"""Debug script to diagnose megakernel KV cache and position handling."""

import torch
from qwen_megakernel.model import Decoder

def main():
    print("=" * 60)
    print("Megakernel Debug Diagnostic")
    print("=" * 60)
    
    dec = Decoder(model_name="Qwen/Qwen3-0.6B", verbose=True)
    
    # Test 1: Single token (like bench.py "Hello")
    print("\n[Test 1] Single token generation (like bench.py)")
    dec.reset()
    tok = dec.step(21806)  # "Hello" token
    print(f"  Input: 21806 ('Hello')")
    print(f"  Output: {tok}")
    print(f"  Position: {dec.position}")
    print(f"  Hidden state mean: {dec._hidden.abs().mean().item():.6f}")
    print(f"  Hidden state std:  {dec._hidden.std().item():.6f}")
    
    # Test 2: Prefill 4 tokens then generate (like compare_tokens.py)
    print("\n[Test 2] Prefill 4 tokens then generate (like compare_tokens.py)")
    dec.reset()
    prompt = [785, 6722, 315, 9625]  # "The capital of France"
    print(f"  Prefilling: {prompt}")
    for i, tid in enumerate(prompt):
        out = dec.step(tid)
        print(f"    Step {i}: input={tid}, output={out}, position={dec.position}")
        if i == len(prompt) - 1:
            print(f"    Hidden state mean: {dec._hidden.abs().mean().item():.6f}")
            print(f"    Hidden state std:  {dec._hidden.std().item():.6f}")
    
    # Generate first token
    tok = dec.step(374)  # " is"
    print(f"  First generated token: {tok} (expected: 12095 ' Paris')")
    print(f"  Position after generation: {dec.position}")
    print(f"  Hidden state mean: {dec._hidden.abs().mean().item():.6f}")
    print(f"  Hidden state std:  {dec._hidden.std().item():.6f}")
    
    # Test 3: Check KV cache contents
    print("\n[Test 3] KV cache inspection")
    print(f"  K cache shape: {dec._k_cache.shape}")
    print(f"  V cache shape: {dec._v_cache.shape}")
    print(f"  K cache [layer=0, head=0, pos=0, :4]: {dec._k_cache[0, 0, 0, :4]}")
    print(f"  K cache [layer=0, head=0, pos=4, :4]: {dec._k_cache[0, 0, 4, :4]}")
    
    # Test 4: Compare with HF directly (matching compare_tokens.py exactly)
    print("\n[Test 4] Comparing with HuggingFace (matching compare_tokens.py)")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    
    # Exact same prompt as compare_tokens.py
    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.cuda()
    attention_mask = torch.ones_like(input_ids)
    
    print(f"  Prompt: {repr(prompt)}")
    print(f"  Prompt IDs: {input_ids[0].tolist()}")
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,  # Just first token
            do_sample=False,
            use_cache=True,
        )
    hf_first_token = output[0, -1].item()
    print(f"  HF first generated token: {hf_first_token} (expected: 12095 ' Paris')")
    print(f"  HF text: {tokenizer.decode([hf_first_token], skip_special_tokens=True)}")
    
    # Test 5: Check if position counter is the issue
    print("\n[Test 5] Position counter check")
    dec2 = Decoder(model_name="Qwen/Qwen3-0.6B", verbose=False)
    dec2.reset()
    prompt_ids = input_ids[0].tolist()
    print(f"  Full prompt IDs: {prompt_ids}")
    print(f"  Prefilling first {len(prompt_ids)-1} tokens...")
    for i, tid in enumerate(prompt_ids[:-1]):
        dec2.step(tid)
        print(f"    After token {i} ({tid}): position={dec2.position}, cache_len should be {dec2.position+1}")
    
    print(f"  Now generating from last prompt token ({prompt_ids[-1]})...")
    print(f"  Position BEFORE step: {dec2.position}")
    print(f"  Expected cache_len in kernel: {dec2.position + 1}")
    first_gen = dec2.step(prompt_ids[-1])
    print(f"  Generated token: {first_gen} (expected: {hf_first_token})")
    print(f"  Position AFTER step: {dec2.position}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
