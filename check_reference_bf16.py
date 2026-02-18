#!/usr/bin/env python3
"""
Run the parity reference prompt in bf16 (and optionally float32) and report
the first generated token. If bf16 gives 1112 instead of 12095, the
megakernel vs HF gap is dtype/numerics; if bf16 still gives 12095, the gap
is likely in the kernel.
"""
import argparse
import sys

PROMPT = "The capital of France is"
EXPECTED_FIRST_TOKEN = 12095  # " Paris" from typical float32 reference
MEGAKERNEL_FIRST = 1112       # "..." — what megakernel gives in bf16


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="Check HF first token in bf16 vs float32")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model name")
    parser.add_argument("--no-float32", action="store_true", help="Skip float32 run (e.g. when no GPU for f32)")
    args = parser.parse_args()
    model_name = args.model

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    enc = tokenizer(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = torch.ones_like(input_ids)
    prompt_ids = input_ids[0].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {model_name}")
    print(f"Prompt: {repr(PROMPT)}")
    print(f"Prompt token IDs: {prompt_ids}")
    print(f"Device: {device}\n")

    results = []

    # --- bf16 (same as megakernel path) ---
    print("--- bf16 (HuggingFace, same dtype as megakernel) ---")
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model_bf16.eval()
    ids_bf16 = input_ids.to(device)
    mask_bf16 = attention_mask.to(device)
    with torch.no_grad():
        out = model_bf16(ids_bf16, attention_mask=mask_bf16)
    logits = out.logits[0, -1, :]
    first_bf16 = logits.argmax().item()
    text_bf16 = tokenizer.decode([first_bf16], skip_special_tokens=False)
    print(f"  First token id: {first_bf16}  ({repr(text_bf16)})")
    results.append(("bf16", first_bf16, text_bf16))
    del model_bf16
    torch.cuda.empty_cache()

    # --- float32 (only on CPU or if user did not skip) ---
    if not args.no_float32:
        print("\n--- float32 (HuggingFace) ---")
        # Run on CPU for true float32; or cuda if you want (then it's still float32)
        f32_device = "cpu" if device == "cuda" else device
        model_f32 = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=f32_device,
            trust_remote_code=True,
        )
        model_f32.eval()
        ids_f32 = input_ids.to(f32_device)
        mask_f32 = attention_mask.to(f32_device)
        with torch.no_grad():
            out = model_f32(ids_f32, attention_mask=mask_f32)
        logits_f32 = out.logits[0, -1, :]
        first_f32 = logits_f32.argmax().item()
        text_f32 = tokenizer.decode([first_f32], skip_special_tokens=False)
        print(f"  First token id: {first_f32}  ({repr(text_f32)})")
        results.append(("float32", first_f32, text_f32))
        del model_f32
        if device == "cuda":
            torch.cuda.empty_cache()

    # --- Summary ---
    print("\n--- Summary ---")
    for dtype, tid, text in results:
        match_exp = " (matches expected 12095)" if tid == EXPECTED_FIRST_TOKEN else ""
        match_mk = " (matches megakernel 1112)" if tid == MEGAKERNEL_FIRST else ""
        print(f"  {dtype}: first token = {tid}{match_exp}{match_mk}")
    first_bf16 = results[0][1]
    if first_bf16 == MEGAKERNEL_FIRST:
        print("\n→ bf16 first token equals megakernel (1112). Gap is dtype/numerics, not a kernel bug.")
    elif first_bf16 == EXPECTED_FIRST_TOKEN:
        print("\n→ bf16 first token equals expected (12095). Gap is likely in the megakernel (e.g. reduction order, RoPE, or lm_head).")
    else:
        print(f"\n→ bf16 first token is {first_bf16} (neither 12095 nor 1112). Re-run with --no-float32 to avoid OOM.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
