"""
compare_tokens.py
=================
Step 2B — Megakernel Parity Check

Loads the ground-truth token IDs produced by parity_reference.py (Mac/CPU),
runs the same prompt through the compiled megakernel (GPU), and reports
whether every token matches exactly.

Usage (must run on the GPU machine with the megakernel compiled)
----------------------------------------------------------------
    python compare_tokens.py
    python compare_tokens.py --ref parity_reference_output.json
    python compare_tokens.py --ref parity_reference_output.json --model Qwen/Qwen3-0.6B

Exit codes
----------
    0  — Parity: TRUE  (all tokens match)
    1  — Parity: FALSE (at least one token differs)
    2  — Setup error   (missing file, compile error, import error)
"""

import argparse
import json
import sys
from pathlib import Path

REFERENCE_FILE = Path(__file__).parent / "parity_reference_output.json"


# ---------------------------------------------------------------------------
# Token comparison logic
# ---------------------------------------------------------------------------

def compare(ref_ids: list[int], mk_ids: list[int], tokenizer) -> bool:
    """
    Print a side-by-side table of reference vs megakernel tokens.
    Returns True if every position matches.
    """
    n = max(len(ref_ids), len(mk_ids))
    all_match = True

    print(f"\n{'pos':>3}  {'ref_id':>7}  {'mk_id':>7}  {'ref_text':<20}  {'mk_text':<20}  match")
    print("─" * 75)

    for i in range(n):
        ref_tok = ref_ids[i] if i < len(ref_ids) else None
        mk_tok  = mk_ids[i]  if i < len(mk_ids)  else None
        match   = ref_tok == mk_tok

        ref_text = tokenizer.decode([ref_tok], skip_special_tokens=False) if ref_tok is not None else "—"
        mk_text  = tokenizer.decode([mk_tok],  skip_special_tokens=False) if mk_tok  is not None else "—"

        flag = "✓" if match else "✗"
        if not match:
            all_match = False

        print(
            f"{i:>3}  {str(ref_tok):>7}  {str(mk_tok):>7}  "
            f"{repr(ref_text):<20}  {repr(mk_text):<20}  {flag}"
        )

    print("─" * 75)
    return all_match


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(ref_path: Path, model_name: str) -> int:
    # ── Load reference ───────────────────────────────────────────────────────
    if not ref_path.exists():
        print(f"[ERROR] Reference file not found: {ref_path}", file=sys.stderr)
        print(
            "Run parity_reference.py on the Mac first, then copy the JSON here.",
            file=sys.stderr,
        )
        return 2

    ref = json.loads(ref_path.read_text())
    ref_ids    = ref["generated_ids"]       # 20 tokens (prompt excluded)
    ref_prompt = ref["prompt"]
    ref_model  = ref["model_name"]
    max_tokens = len(ref_ids)

    print("=" * 65)
    print("Megakernel Parity Check")
    print("=" * 65)
    print(f"Reference file : {ref_path}")
    print(f"Reference model: {ref_model}")
    print(f"Prompt         : {repr(ref_prompt)}")
    print(f"Tokens to match: {max_tokens}")
    print()

    if model_name != ref_model:
        print(
            f"  ⚠  --model '{model_name}' differs from reference model '{ref_model}'.\n"
            f"     Parity is only meaningful when both use the same checkpoint.\n"
        )

    # ── Import megakernel ────────────────────────────────────────────────────
    try:
        from qwen_megakernel.model import Decoder
    except ImportError as e:
        print(f"[ERROR] Could not import megakernel: {e}", file=sys.stderr)
        print(
            "\nThe megakernel must be compiled before running this script:\n"
            "  pip install ninja && python qwen_megakernel/build.py\n"
            "This script must run on the GPU machine (CUDA required).",
            file=sys.stderr,
        )
        return 2

    # ── Load model ───────────────────────────────────────────────────────────
    print("Loading megakernel decoder...")
    try:
        dec = Decoder(model_name=model_name, verbose=True)
    except Exception as e:
        print(f"[ERROR] Decoder init failed: {e}", file=sys.stderr)
        return 2

    tokenizer = dec.tokenizer

    # ── Encode prompt and prefill ─────────────────────────────────────────────
    # Replicate exactly what parity_reference.py does:
    #   tokenizer(prompt) → prompt_ids
    #   step() each prompt token except the last (prefill)
    #   then step() the last prompt token → first generated token
    prompt_ids = tokenizer.encode(ref_prompt, add_special_tokens=True)

    print(f"Prompt token IDs ({len(prompt_ids)} tokens): {prompt_ids}")
    print(f"Reference prompt IDs                       : {ref['prompt_ids']}")

    if prompt_ids != ref["prompt_ids"]:
        print(
            "\n  ✗  TOKENIZER MISMATCH — prompt encodes differently.\n"
            "     This will cause all tokens to diverge regardless of kernel correctness.\n"
            "     Ensure both machines use the same tokenizer checkpoint.\n"
        )

    # Prefill all but the last prompt token
    dec.reset()
    for tid in prompt_ids[:-1]:
        dec.step(tid)

    # ── Generate max_tokens using single-step decode ──────────────────────────
    # Using step() (not generate()) so we collect each individual token id
    # before EOS pruning — matching exactly what the reference script captures.
    import torch
    print(f"\nRunning megakernel for {max_tokens} tokens...")
    mk_ids = []
    tok = prompt_ids[-1]
    with torch.no_grad():
        for _ in range(max_tokens):
            tok = dec.step(tok)
            mk_ids.append(tok)

    # ── Compare ───────────────────────────────────────────────────────────────
    all_match = compare(ref_ids, mk_ids, tokenizer)

    print(f"\nReference : {ref_ids}")
    print(f"Megakernel: {mk_ids}")

    ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)
    mk_text  = tokenizer.decode(mk_ids,  skip_special_tokens=True)
    print(f"\nReference text : {repr(ref_text)}")
    print(f"Megakernel text: {repr(mk_text)}")

    print()
    if all_match:
        print("══════════════════════════════════════")
        print("  Parity: TRUE  ✓  All tokens match")
        print("══════════════════════════════════════")
        return 0
    else:
        mismatches = sum(
            1 for r, m in zip(ref_ids, mk_ids) if r != m
        ) + abs(len(ref_ids) - len(mk_ids))
        print("══════════════════════════════════════════════════════════")
        print(f"  Parity: FALSE  ✗  {mismatches}/{max_tokens} token(s) differ")
        print("══════════════════════════════════════════════════════════")
        print("\nCommon causes:")
        print("  1. rope_theta mismatch  — check model.py inv_freq base")
        print("  2. lm_head not loaded   — ensure state['lm_head.weight'] is used")
        print("  3. dtype mismatch       — verify all weights are bf16 on GPU")
        print("  4. KV cache not reset   — dec.reset() must be called before each run")
        return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare megakernel token output against CPU reference."
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=REFERENCE_FILE,
        help=f"Path to parity_reference_output.json (default: {REFERENCE_FILE})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model id (default: read from reference JSON)",
    )
    args = parser.parse_args()

    # Default model name to whatever the reference was generated with
    if args.model is None:
        try:
            ref_data = json.loads(args.ref.read_text())
            args.model = ref_data["model_name"]
        except Exception:
            args.model = "Qwen/Qwen3-0.6B"

    sys.exit(main(args.ref, args.model))
