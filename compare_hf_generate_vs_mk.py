#!/usr/bin/env python3
"""
Compare HF generate() (incremental, with KV cache) vs megakernel step-by-step.
Uses the same prompt; runs HF with max_new_tokens=1 repeatedly (true incremental)
and MK with step() in a loop. Reports first position where they differ.
Run on GPU.
"""
import sys

PROMPT = "The capital of France is"
MODEL_NAME = "Qwen/Qwen3-0.6B"
NUM_TOKENS = 20


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=True)

    print("HF generate (incremental, use_cache=True) vs Megakernel step-by-step")
    print("=" * 60)
    print(f"Prompt: {repr(PROMPT)}")
    print(f"Prompt IDs: {prompt_ids}\n")

    # Load HF
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    # HF: true incremental (like parity_reference) - generate one token at a time with cache
    print("HF incremental (generate max_new_tokens=1 in a loop)...")
    input_ids = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)
    hf_ids = []
    with torch.no_grad():
        for _ in range(NUM_TOKENS):
            out = model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            next_tok = out[0, -1].item()
            hf_ids.append(next_tok)
            input_ids = out  # keep full sequence for next iteration

    # MK: step-by-step
    print("Megakernel step-by-step...")
    from qwen_megakernel.model import Decoder
    dec = Decoder(model_name=MODEL_NAME, verbose=False)
    dec.reset()
    for t in prompt_ids[:-1]:
        dec.step(t)
    tok = prompt_ids[-1]
    mk_ids = []
    with torch.no_grad():
        for _ in range(NUM_TOKENS):
            tok = dec.step(tok)
            mk_ids.append(tok)

    del model
    torch.cuda.empty_cache()

    # Compare
    print()
    first_diff = None
    for i in range(NUM_TOKENS):
        match = hf_ids[i] == mk_ids[i]
        sym = "✓" if match else "✗"
        if not match and first_diff is None:
            first_diff = i
        print(f"  [{i:2d}]  HF={hf_ids[i]:6d}  MK={mk_ids[i]:6d}  {sym}")
    print()
    if first_diff is not None:
        print(f"First mismatch at index {first_diff}: HF={hf_ids[first_diff]} ({tokenizer.decode([hf_ids[first_diff]])})  MK={mk_ids[first_diff]} ({tokenizer.decode([mk_ids[first_diff]])})")
        print("→ Megakernel does not match HF generate() at this position; kernel fix needed.")
    else:
        print("Parity: HF generate() and MK match for all tokens.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
