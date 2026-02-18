"""
parity_reference.py
====================
Step 2A — Deterministic Reference Script

Purpose
-------
Generates a ground-truth token sequence using HuggingFace transformers for
a given prompt.  The output is saved to a JSON file so it can later be
compared against the megakernel's output on a GPU machine (Step 2B).

Generation rules
----------------
  - do_sample=False              greedy, fully deterministic
  - temperature=1.0              no effect under greedy; stated explicitly
  - top_p / top_k disabled
  - repetition_penalty=1.0
  - attention_mask explicitly set to all-ones (no padding ambiguity)
  - max_new_tokens=20
  - Prompt: "The capital of France is"

Known issues caught by this script
-----------------------------------
  1. Attention mask  — must be passed explicitly; omitting it causes a
     HuggingFace warning and non-deterministic padding behaviour.

  2. Embedding tie  — even when tie_word_embeddings=True in config,
     both embed_tokens.weight and lm_head.weight may exist as separate
     tensors in the state dict (transformers 5.x behaviour).
     The megakernel's load_weights() must load lm_head.weight explicitly.

  3. RoPE theta     — Qwen3-0.6B config.rope_theta = 10000 but
     config.rope_scaling['rope_theta'] = 1000000.  The effective base
     used by HuggingFace is the rope_scaling value (1000000).
     The megakernel hardcodes 10000 in model.py — this MUST be fixed
     before GPU parity testing.

Usage
-----
    python parity_reference.py
    python parity_reference.py --model Qwen/Qwen3-0.6B
    python parity_reference.py --model /local/path/to/weights

Output file: parity_reference_output.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Correct HuggingFace model ID for the Qwen3-TTS 0.6B talker backbone.
# Use --model Qwen/Qwen3-0.6B for a quick smoke-test without the TTS weights.
DEFAULT_MODEL  = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
PROMPT         = "The capital of France is"
MAX_NEW_TOKENS = 20
OUTPUT_FILE    = Path(__file__).resolve().parent / "parity_reference_output.json"

# Megakernel compile-time constants — used for config cross-check only.
MK_HIDDEN_SIZE        = 1024
MK_NUM_LAYERS         = 28
MK_NUM_KV_HEADS       = 8
MK_HEAD_DIM           = 128
MK_VOCAB_SIZE         = 151936
MK_ROPE_THETA         = 1000000.0  # model.py reads config.rope_theta dynamically
MK_MAX_SEQ_LEN        = 2048

# ---------------------------------------------------------------------------
# Device / dtype helpers
# ---------------------------------------------------------------------------

def _pick_device() -> str:
    """Prefer CUDA, then MPS, then CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _safe_dtype(device: str):
    """
    Match the runtime numerics as closely as practical:
      - CUDA: bfloat16 (same as megakernel weight path)
      - CPU/MPS: float32 fallback
    """
    import torch
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


# ---------------------------------------------------------------------------
# qwen3_tts architecture registration
# ---------------------------------------------------------------------------

def _ensure_qwen_tts_registered() -> bool:
    """
    Qwen3-TTS-12Hz models use architecture 'qwen3_tts' which is NOT built
    into transformers core.  Importing the qwen_tts package registers it
    so that AutoConfig / AutoModelForCausalLM work as normal.

    Returns True if available, False if not installed.
    """
    try:
        import qwen_tts  # noqa: F401 — registers qwen3_tts auto-class hooks
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Config cross-check
# ---------------------------------------------------------------------------

def _check_config(config) -> dict:
    """
    Extract model config fields and cross-check against megakernel constants.
    Prints a labelled warning for every mismatch.  Does NOT raise.
    Returns a plain dict suitable for JSON serialisation.
    """
    checks = {
        "tied_embeddings": getattr(config, "tie_word_embeddings", None),
        "rope_theta":      float(getattr(config, "rope_theta", 10000.0)),
        "rope_scaling":    getattr(config, "rope_scaling",  None),
        "vocab_size":      getattr(config, "vocab_size",    None),
        "hidden_size":     getattr(config, "hidden_size",   None),
        "num_layers":      getattr(config, "num_hidden_layers", None),
        "num_kv_heads":    getattr(config, "num_key_value_heads", None),
        "head_dim":        getattr(config, "head_dim", None),
    }

    # Derive head_dim if absent from config
    if checks["head_dim"] is None:
        num_q = getattr(config, "num_attention_heads", None)
        if num_q and checks["hidden_size"]:
            checks["head_dim"] = checks["hidden_size"] // num_q

    # Resolve effective rope_theta: rope_scaling dict may override config field
    effective_rope_theta = checks["rope_theta"]
    if isinstance(checks["rope_scaling"], dict):
        effective_rope_theta = float(
            checks["rope_scaling"].get("rope_theta", checks["rope_theta"])
        )
    checks["effective_rope_theta"] = effective_rope_theta

    print("\n── Config cross-check (vs megakernel constants) ──────────────────")

    # tie_word_embeddings
    if checks["tied_embeddings"] is False:
        print("  ⚠  tie_word_embeddings = False")
        print("     load_weights() must load lm_head.weight as a separate tensor.")
    else:
        print(f"  ✓  tie_word_embeddings = {checks['tied_embeddings']}")
        print("     Even so — lm_head.weight may exist separately (see embedding check).")

    # rope_theta vs effective_rope_theta
    if abs(effective_rope_theta - MK_ROPE_THETA) > 1.0:
        print(f"  ✗  effective rope_theta = {effective_rope_theta}  "
              f"(megakernel hardcodes {MK_ROPE_THETA})")
        print("     ACTION REQUIRED: update MK_ROPE_THETA in parity_reference.py")
        print(f"       or verify model.py reads config.rope_theta dynamically.")
    else:
        print(f"  ✓  rope_theta = {effective_rope_theta}")

    # rope_scaling type
    rs = checks["rope_scaling"]
    if rs is not None:
        rs_type = rs.get("rope_type", "unknown") if isinstance(rs, dict) else rs
        if rs_type not in ("default", None):
            print(f"  ⚠  rope_scaling.rope_type = '{rs_type}'")
            print("     Megakernel has no position scaling — RoPE tables must be rebuilt.")
        else:
            print(f"  ✓  rope_scaling.rope_type = '{rs_type}' (no position scaling)")
    else:
        print("  ✓  rope_scaling = None")

    # vocab_size
    if checks["vocab_size"] and checks["vocab_size"] != MK_VOCAB_SIZE:
        print(f"  ✗  vocab_size = {checks['vocab_size']}  (megakernel = {MK_VOCAB_SIZE})")
        print("     LDG_VOCAB_SIZE in kernel.cu and MK_VOCAB_SIZE here must be updated.")
    else:
        print(f"  ✓  vocab_size = {checks['vocab_size']}")

    # hidden_size
    if checks["hidden_size"] and checks["hidden_size"] != MK_HIDDEN_SIZE:
        print(f"  ✗  hidden_size = {checks['hidden_size']}  (megakernel = {MK_HIDDEN_SIZE})")
    else:
        print(f"  ✓  hidden_size = {checks['hidden_size']}")

    # num_layers
    if checks["num_layers"] and checks["num_layers"] != MK_NUM_LAYERS:
        print(f"  ✗  num_hidden_layers = {checks['num_layers']}  (megakernel = {MK_NUM_LAYERS})")
    else:
        print(f"  ✓  num_hidden_layers = {checks['num_layers']}")

    # num_kv_heads
    if checks["num_kv_heads"] and checks["num_kv_heads"] != MK_NUM_KV_HEADS:
        print(f"  ✗  num_key_value_heads = {checks['num_kv_heads']}  (megakernel = {MK_NUM_KV_HEADS})")
    else:
        print(f"  ✓  num_key_value_heads = {checks['num_kv_heads']}")

    # head_dim
    if checks["head_dim"] and checks["head_dim"] != MK_HEAD_DIM:
        print(f"  ✗  head_dim = {checks['head_dim']}  (megakernel = {MK_HEAD_DIM})")
    else:
        print(f"  ✓  head_dim = {checks['head_dim']}")

    print("──────────────────────────────────────────────────────────────────\n")
    return checks


# ---------------------------------------------------------------------------
# Embedding tie inspection
# ---------------------------------------------------------------------------

def _check_embeddings(model) -> dict:
    """
    Inspect actual tensor identity of embed_tokens.weight vs lm_head.weight.

    Checks three distinct scenarios:
      A. lm_head.weight absent from state_dict  → truly tied (weight shared)
      B. Both present, same data_ptr            → tied in memory
      C. Both present, different data_ptr       → untied in memory
         (transformers 5.x may create separate tensors even when config says tied)

    Returns a dict with shape and tie status for the JSON output.
    The megakernel's load_weights() must explicitly load lm_head.weight
    in cases B and C to guarantee it uses the correct tensor.
    """
    import torch
    state       = model.state_dict()
    has_lm_head = "lm_head.weight" in state
    has_embed   = "model.embed_tokens.weight" in state

    result = {
        "lm_head_in_state_dict": has_lm_head,
        "embed_in_state_dict":   has_embed,
        "truly_tied":            None,
        "lm_head_shape":         None,
        "embed_shape":           None,
        "lm_head_dtype":         None,
    }

    print("── Embedding tensor inspection ───────────────────────────────────")

    if has_embed:
        emb = state["model.embed_tokens.weight"]
        result["embed_shape"] = list(emb.shape)
        print(f"  embed_tokens.weight  shape={tuple(emb.shape)}  dtype={emb.dtype}")

    if not has_lm_head:
        print("  lm_head.weight  → NOT in state_dict (weight shared via embed)")
        print("  ✓  Truly tied — load_weights() embed alias is safe here.")
        result["truly_tied"] = True
    else:
        lm = state["lm_head.weight"]
        result["lm_head_shape"] = list(lm.shape)
        result["lm_head_dtype"] = str(lm.dtype)
        print(f"  lm_head.weight       shape={tuple(lm.shape)}  dtype={lm.dtype}")

        if has_embed:
            same_ptr    = lm.data_ptr() == emb.data_ptr()
            same_values = torch.equal(lm, emb)

            if same_ptr:
                result["truly_tied"] = True
                print("  ✓  Same data_ptr — tied in memory.")
            elif same_values:
                result["truly_tied"] = False
                print("  ⚠  Different data_ptr, equal values — COPIED not tied.")
                print("     (transformers 5.x behaviour with tie_word_embeddings=True)")
                print("     ACTION: load_weights() must use state['lm_head.weight'] explicitly.")
            else:
                result["truly_tied"] = False
                print("  ✗  UNTIED — lm_head.weight differs from embed_tokens.weight.")
                print("     ACTION: load_weights() must use state['lm_head.weight'] explicitly.")

    print("──────────────────────────────────────────────────────────────────\n")
    return result


# ---------------------------------------------------------------------------
# Main generate function
# ---------------------------------------------------------------------------

def run(model_name: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    device = _pick_device()
    dtype  = _safe_dtype(device)

    print(f"Model  : {model_name}")
    print(f"Device : {device}")
    print(f"Dtype  : {dtype}")
    print(f"Prompt : {repr(PROMPT)}\n")

    # ── Register qwen3_tts model type if needed ──────────────────────────────
    qwen_tts_ok = _ensure_qwen_tts_registered()
    if not qwen_tts_ok:
        print(
            "  ℹ  qwen_tts package not found.\n"
            "     Qwen3-TTS-12Hz models need:  pip install qwen-tts\n"
            "     Qwen/Qwen3-0.6B works without it.\n"
        )

    # ── Config ───────────────────────────────────────────────────────────────
    print("Loading config...")
    config     = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_type = getattr(config, "model_type", "unknown")
    print(f"Architecture : {model_type}")
    config_checks = _check_config(config)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # ── Tokenize and build attention mask ────────────────────────────────────
    enc = tokenizer(PROMPT, return_tensors="pt").to(device)
    input_ids      = enc["input_ids"]
    # FIX 1: Explicit attention mask — all ones, no padding.
    # Omitting this causes a HuggingFace warning and non-deterministic padding
    # handling; the CUDA kernel will assume strict left-aligned inputs.
    attention_mask = torch.ones_like(input_ids)

    prompt_ids = input_ids[0].tolist()
    print(f"Prompt token IDs ({len(prompt_ids)} tokens): {prompt_ids}")
    print(f"Attention mask : {attention_mask[0].tolist()}\n")

    total_len = len(prompt_ids) + MAX_NEW_TOKENS
    if total_len > MK_MAX_SEQ_LEN:
        print(f"  ⚠  prompt ({len(prompt_ids)}) + max_new_tokens ({MAX_NEW_TOKENS}) "
              f"= {total_len} > MK_MAX_SEQ_LEN ({MK_MAX_SEQ_LEN})")

    # ── Load model ───────────────────────────────────────────────────────────
    # Use torch_dtype= for compatibility with transformers 4.57.x (required by qwen-tts).
    print(f"Loading model weights (device={device}, dtype={dtype})...")
    t0    = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s\n")

    # ── Embedding tie inspection ──────────────────────────────────────────────
    embed_checks = _check_embeddings(model)

    # ── Greedy generate ───────────────────────────────────────────────────────
    print(f"Generating {MAX_NEW_TOKENS} tokens (greedy, deterministic)...")
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,   # FIX 1 applied here
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_ids      = out[0].tolist()
    generated_ids = full_ids[len(prompt_ids):]

    print(f"\n── Generated token IDs (ground truth) ──────────────────────────")
    for i, tok in enumerate(generated_ids):
        text = tokenizer.decode([tok], skip_special_tokens=False)
        print(f"  [{i:02d}]  id={tok:6d}   '{text}'")
    print("──────────────────────────────────────────────────────────────────")

    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\nDecoded : {repr(decoded)}\n")

    # ── Serialise config_checks for JSON ─────────────────────────────────────
    for k, v in config_checks.items():
        if not isinstance(v, (str, int, float, bool, type(None))):
            config_checks[k] = str(v)

    return {
        "model_name":    model_name,
        "model_type":    model_type,
        "device":        device,
        "weight_dtype":  str(dtype),
        "prompt":        PROMPT,
        "prompt_ids":    prompt_ids,
        "generated_ids": generated_ids,
        "full_ids":      full_ids,
        "decoded_text":  decoded,
        "config_checks": config_checks,
        "embed_checks":  embed_checks,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate deterministic reference token IDs for parity testing."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model id or local path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON path (default: parity_reference_output.json next to this script)",
    )
    args = parser.parse_args()

    out_path = Path(args.output).resolve() if args.output else OUTPUT_FILE
    print(f"Output file: {out_path}\n")

    try:
        result = run(args.model)

    except OSError as e:
        print(f"\n[ERROR] Could not load model '{args.model}':\n  {e}", file=sys.stderr)
        print(
            "\nTroubleshooting:\n"
            "  Talker backbone : python parity_reference.py\n"
            "  Base model      : python parity_reference.py --model Qwen/Qwen3-0.6B\n"
            "  Local weights   : python parity_reference.py --model /path/to/weights\n",
            file=sys.stderr,
        )
        sys.exit(1)

    except ValueError as e:
        if "qwen3_tts" in str(e).lower() or "unknown model type" in str(e).lower():
            print(f"\n[ERROR] Unknown model architecture: {e}", file=sys.stderr)
            print(
                "\nThe Qwen3-TTS-12Hz model requires the qwen-tts package:\n"
                "  pip install qwen-tts\n"
                "Then re-run.  Or smoke-test with:\n"
                "  python parity_reference.py --model Qwen/Qwen3-0.6B\n",
                file=sys.stderr,
            )
        else:
            print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    try:
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Ground truth saved → {out_path}")
        print("Next: copy parity_reference_output.json to GPU machine for Step 2B.")
    except OSError as e:
        print(f"\n[ERROR] Could not write output to {out_path}:\n  {e}", file=sys.stderr)
        sys.exit(1)
