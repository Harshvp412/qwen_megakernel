# First-token parity debug guide

## Status

- **RoPE theta**: Confirmed 1000000.0 in kernel (cos/sin tables). Matches Qwen3-0.6B.
- **Position**: Prefill uses positions 0..N-1; first decode step uses position N-1 for RoPE and KV write. Matches HF (last-token position_id = N-1).
- **LM head**: Step() uses logits+argmax path (not fused argmax) so the chosen token matches HF. First token and first 5 generated tokens now match.
- **Current parity**: First 5 generated tokens match; first mismatch at generated token index 5 (reference 9625 " France", megakernel 15344 " Italy"). Divergence then grows. Likely cause: KV cache write/read or attention accumulation once cache has 10+ positions; or bf16 drift over steps.

## Verified in code

1. **RoPE**
   - `inv_freq = 1 / (rope_theta ** (arange(0, 128, 2) / 128))` — matches HF `compute_default_rope_parameters`.
   - Cos/sin: `freqs = outer(positions, inv_freq)`, then `cos(freqs).repeat(1,2)` — same layout as HF.
   - Kernel applies `(x0*cos - x1*sin, x0*sin + x1*cos)` per pair — matches `apply_rotary_pos_emb`.

2. **Weights**
   - `lm_head.weight` loaded when present (no alias to embed when untied).
   - LM head kernel: `weight[vocab_id, :]` dot `hidden` → row-major (vocab_size, hidden_size), correct.

3. **Flow**
   - Prefill: step(ids[0]), step(ids[1]), … step(ids[N-2]) → KV at 0..N-2, position = N-1.
   - First gen: step(ids[N-1]) with position N-1 → RoPE at N-1, KV written at N-1, cache_len = N, attend over 0..N-1.
   - So we attend over N positions (0..N-1) when producing the first token. HF does the same (one forward with N tokens, logits at last position).

## How to debug

### 1. Run logits comparison (GPU)

```bash
python debug_parity_logits.py
```

- Prints HF top-10 next-token logits and megakernel’s first token.
- If megakernel token is **not** in HF top-10 → big divergence (wrong hidden state or wrong lm_head path).
- If it **is** in top-10 → smaller numerical/ordering difference (e.g. bf16, or argmax tie).

### 2. Regenerate reference on same machine

Reference may have been generated with different tokenizer/model revision/dtype:

```bash
python parity_reference.py --model Qwen/Qwen3-0.6B
python compare_tokens.py
```

Use the same env (transformers version, tokenizer) as when you run the megakernel.

### 3. Confirm prompt_ids match

In `compare_tokens.py` output, check:

- `Prompt token IDs (5 tokens): [...]`
- `Reference prompt IDs: [...]`

If these differ, tokenizer or prompt encoding differs → regenerate reference on this machine.

## Possible root causes (if RoPE/position/weights are correct)

1. **Numerical**: bf16 in kernel vs float32 in reference → can change argmax; usually only first token or a few positions.
2. **Kernel bug** in one of: RMSNorm, attention (online softmax, reduction), MLP, or final norm.
3. **State dict / layout**: Wrong key, transposed tensor, or different layer order (e.g. q_norm/k_norm after proj) vs HF.
4. **Reference vs runner**: Different model revision, tokenizer, or `add_special_tokens` → regenerate reference on GPU machine.

## Next steps

- **Find first divergence**: `python debug_step_logits.py` — teacher-forced run, compares next-token logits at each gen_step; first step where argmax differs is where the bug is introduced.
- **Kernel fix**: In `ldg_attention`, block 0 now does `__threadfence()` (all threads) before signaling `kv_flag`, so KV cache write is visible to all blocks before any attention read.
- **Reference**: `parity_reference_output.json` was restored to the correct HF sequence (9625 at index 5). Regenerate with `python parity_reference.py --model Qwen/Qwen3-0.6B` if needed.
- **Logits**: `python compare_logits.py` compares first-token logits (HF vs MK).
