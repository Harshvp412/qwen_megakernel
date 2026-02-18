#!/usr/bin/env python3
"""
Compare first-token logits: HuggingFace (bf16) vs megakernel.
Run on GPU. Prints argmax, top-10, and logit values for 12095/1112 to diagnose
why megakernel picks 1112 instead of 12095.
"""
import sys

PROMPT = "The capital of France is"
MODEL_NAME = "Qwen/Qwen3-0.6B"
EXPECTED_ID = 12095  # " Paris"
MEGAKERNEL_ID = 1112  # "..."


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("Compare first-token logits: HF vs megakernel")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {repr(PROMPT)}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=True)
    print(f"Prompt token IDs: {prompt_ids}\n")

    # --- HuggingFace logits (bf16) ---
    print("--- HuggingFace (bf16) ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    input_ids = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    hf_logits = out.logits[0, -1, :].float()  # [vocab_size], float32 for comparison
    hf_argmax = hf_logits.argmax().item()
    print(f"Argmax: {hf_argmax} ({tokenizer.decode([hf_argmax], skip_special_tokens=False)})")
    top10 = torch.topk(hf_logits, 10)
    print("Top-10:", list(zip(top10.indices.cpu().tolist(), top10.values.cpu().tolist())))
    del model
    torch.cuda.empty_cache()

    # --- Megakernel logits ---
    print("\n--- Megakernel ---")
    from qwen_megakernel.model import Decoder

    dec = Decoder(model_name=MODEL_NAME, verbose=False)
    dec.reset()
    for tid in prompt_ids[:-1]:
        dec.step(tid)
    mk_token, mk_logits = dec.step(prompt_ids[-1], return_logits=True)
    mk_logits = mk_logits.cpu()  # already float32
    mk_argmax = mk_logits.argmax().item()
    print(f"Argmax: {mk_argmax} ({tokenizer.decode([mk_argmax], skip_special_tokens=False)})")
    mk_top10 = torch.topk(mk_logits, 10)
    print("Top-10:", list(zip(mk_top10.indices.tolist(), mk_top10.values.tolist())))

    # --- Compare key tokens ---
    print("\n--- Logit comparison (key tokens) ---")
    hf_l = hf_logits.cpu()
    for tid, name in [(EXPECTED_ID, "12095 ( Paris)"), (MEGAKERNEL_ID, "1112 (...)")]:
        a = hf_l[tid].item()
        b = mk_logits[tid].item()
        diff = b - a
        print(f"  {name}:  HF={a:.4f}  MK={b:.4f}  diff(MK-HF)={diff:.4f}")

    # --- Summary ---
    print("\n--- Summary ---")
    print(f"  HF  argmax: {hf_argmax}  MK argmax: {mk_argmax}")
    if hf_argmax != mk_argmax:
        print(f"  Logit at 12095: HF={hf_l[EXPECTED_ID]:.4f}  MK={mk_logits[EXPECTED_ID]:.4f}")
        print(f"  Logit at 1112:  HF={hf_l[MEGAKERNEL_ID]:.4f}  MK={mk_logits[MEGAKERNEL_ID]:.4f}")
        if mk_logits[EXPECTED_ID] > mk_logits[MEGAKERNEL_ID]:
            print("  MK would prefer 12095 over 1112 by logits; check for numerical/ordering bug in kernel.")
        else:
            print("  MK logits prefer 1112 over 12095 -> hidden state or lm_head path differs from HF.")
    else:
        print("  First-token parity: match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
