#!/usr/bin/env python3
"""
Find the first generation step where megakernel logits diverge from HF.
Uses teacher forcing: we feed the reference tokens and compare next-token
logits at each step. Run on GPU.
"""
import sys

PROMPT = "The capital of France is"
MODEL_NAME = "Qwen/Qwen3-0.6B"
# Correct HF reference (Paris/France); repo JSON may have been overwritten with MK output
REF_GENERATED = [
    12095, 13, 576, 6722, 315, 9625, 374, 1083, 279, 6722, 315, 279, 5429, 315,
    9625, 13, 576, 6722, 315, 9625,
]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=True)

    print("=" * 60)
    print("Step-by-step logits: find first divergence (teacher-forced)")
    print("=" * 60)
    print(f"Prompt: {repr(PROMPT)}")
    print(f"Prompt IDs: {prompt_ids}")
    print(f"Reference next tokens (first 10): {REF_GENERATED[:10]}\n")

    # Load HF once
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    from qwen_megakernel.model import Decoder
    dec = Decoder(model_name=MODEL_NAME, verbose=False)

    max_steps = min(6, len(REF_GENERATED))  # 6 steps enough to see divergence at 5
    first_divergence = None

    for gen_step in range(max_steps):
        print(f"Step {gen_step}: HF forward...", flush=True)
        # HF: input = prompt + ref[:gen_step], logits at last position
        input_ids = prompt_ids + REF_GENERATED[:gen_step]
        with torch.no_grad():
            out = model(
                torch.tensor([input_ids], device="cuda", dtype=torch.long),
                attention_mask=torch.ones(1, len(input_ids), device="cuda"),
            )
        hf_logits = out.logits[0, -1, :].float()
        hf_argmax = hf_logits.argmax().item()
        expected = REF_GENERATED[gen_step] if gen_step < len(REF_GENERATED) else None

        print(f"Step {gen_step}: MK run...", flush=True)
        # MK: prefill; then step(last_prompt), step(ref[0]), ..., step(ref[gen_step-1]); last step returns logits
        dec.reset()
        for t in prompt_ids[:-1]:
            dec.step(t)
        if gen_step == 0:
            _, mk_logits = dec.step(prompt_ids[-1], return_logits=True)
            mk_logits = mk_logits.cpu()
        else:
            dec.step(prompt_ids[-1])
            for i in range(gen_step - 1):
                dec.step(REF_GENERATED[i])
            print(f"Step {gen_step}: MK last step (position={dec.position}, return_logits)...", flush=True)
            _, mk_logits = dec.step(REF_GENERATED[gen_step - 1], return_logits=True)
            mk_logits = mk_logits.cpu()
        print(f"Step {gen_step}: MK done.", flush=True)

        mk_argmax = mk_logits.argmax().item()
        match = hf_argmax == mk_argmax
        if not match and first_divergence is None:
            first_divergence = gen_step

        hf_top3 = hf_logits.topk(3)
        mk_top3 = mk_logits.topk(3)
        status = "✓" if match else "✗ DIVERGED"
        print(f"Step {gen_step:2d}:  HF argmax={hf_argmax:6d}  MK argmax={mk_argmax:6d}  "
              f"expected={expected}  {status}", flush=True)
        if not match:
            print(f"         HF top3: {[(int(hf_top3.indices[i]), float(hf_top3.values[i])) for i in range(3)]}", flush=True)
            print(f"         MK top3: {[(int(mk_top3.indices[i]), float(mk_top3.values[i])) for i in range(3)]}", flush=True)
            # Logit at expected token
            if expected is not None:
                print(f"         Logit at expected {expected}: HF={hf_logits[expected]:.4f}  MK={mk_logits[expected]:.4f}", flush=True)

    del model
    torch.cuda.empty_cache()

    print(flush=True)
    if first_divergence is not None:
        print(f"First divergence at gen_step {first_divergence} (next token index {first_divergence}).", flush=True)
        print(f"  → Bug is introduced when processing token at position {len(prompt_ids) + first_divergence - 1}", flush=True)
        print(f"  → Check KV write at that position or attention over 0..{len(prompt_ids) + first_divergence - 1}", flush=True)
    else:
        print("No divergence in the steps checked.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
