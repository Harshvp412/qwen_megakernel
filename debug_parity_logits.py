#!/usr/bin/env python3
"""
Debug first-token parity: compare HuggingFace logits vs megakernel output.
Run on GPU machine. Prints HF top-10 next-token logits and megakernel's token
to see if the kernel output is in the ballpark or completely wrong.
"""
import sys
from pathlib import Path

def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-0.6B"
    prompt = "The capital of France is"
    # Expected first token from reference: 12095 (" Paris")
    expected_first = 12095

    print("=" * 60)
    print("Debug: First-token logits (HF vs megakernel)")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Prompt: {repr(prompt)}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"Prompt token IDs ({len(prompt_ids)}): {prompt_ids}")

    # --- HuggingFace: one forward, get logits for next token ---
    print("\n--- HuggingFace ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    input_ids = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    # Confirm RoPE config (should match megakernel's 1000000 for parity)
    rtheta = getattr(model.config, "rope_theta", None)
    rscaling = getattr(model.config, "rope_scaling", None)
    rparams = getattr(model.config, "rope_parameters", None)
    print(f"HF config: rope_theta={rtheta}, rope_scaling={rscaling}, rope_parameters={getattr(rparams, '__dict__', rparams)}")

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[0, -1, :]  # [vocab_size]
    probs = torch.softmax(logits.float(), dim=-1)
    topk = torch.topk(probs, 10)
    print("Top-10 next tokens (HF):")
    for i, (idx, p) in enumerate(zip(topk.indices.cpu().tolist(), topk.values.cpu().tolist())):
        tok = tokenizer.decode([idx], skip_special_tokens=False)
        mark = " <-- expected" if idx == expected_first else ""
        print(f"  {i+1}. id={idx:6d}  p={p:.5f}  {repr(tok)}{mark}")
    hf_argmax = logits.argmax().item()
    print(f"HF argmax token: {hf_argmax} ({tokenizer.decode([hf_argmax], skip_special_tokens=False)})")
    del model
    torch.cuda.empty_cache()

    # --- Megakernel ---
    print("\n--- Megakernel ---")
    try:
        from qwen_megakernel.model import Decoder
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run on GPU machine with megakernel built.")
        return 1

    dec = Decoder(model_name=model_name, verbose=False)
    dec.reset()
    for tid in prompt_ids[:-1]:
        dec.step(tid)
    # No override: use position=4 for RoPE (matches HF's last-token position in full forward)
    mk_token = dec.step(prompt_ids[-1])
    print(f"Megakernel first token (no override): {mk_token} ({tokenizer.decode([mk_token], skip_special_tokens=False)})")
    # With override=5: often worse (e.g. 32671); with no override we get 1112 (HF #4)
    dec.reset()
    for tid in prompt_ids[:-1]:
        dec.step(tid)
    mk_token_override = dec.step(prompt_ids[-1], rope_position_override=len(prompt_ids))
    print(f"Megakernel first token (override=len(prompt)): {mk_token_override} ({tokenizer.decode([mk_token_override], skip_special_tokens=False)})")

    # --- Compare ---
    hf_top_ids = topk.indices.cpu().tolist()
    print("\n--- Comparison ---")
    for label, mid in [("no override", mk_token), ("override=len(prompt)", mk_token_override)]:
        if mid == hf_argmax:
            print(f"Megakernel ({label}): MATCHES HF argmax.")
        else:
            rank = None
            for i, idx in enumerate(hf_top_ids):
                if idx == mid:
                    rank = i + 1
                    break
            if rank is not None:
                print(f"Megakernel ({label}): {mid} is HF's #{rank} choice.")
            else:
                print(f"Megakernel ({label}): {mid} is NOT in HF top-10.")
    if mk_token != hf_argmax and mk_token_override != hf_argmax:
        print(f"Neither matches HF argmax {hf_argmax} ( Paris). Likely bf16 or small kernel bug.")
    print()
    return 0

if __name__ == "__main__":
    sys.exit(main())
