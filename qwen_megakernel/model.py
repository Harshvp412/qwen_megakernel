"""Weight loading and high-level decode API for Qwen3-0.6B."""

from __future__ import annotations

import math
import struct
from typing import Optional, Tuple, Union

import torch

NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
Q_SIZE = 16 * HEAD_DIM  # 2048
KV_SIZE = 8 * HEAD_DIM  # 1024
MAX_SEQ_LEN = 2048
VOCAB_SIZE = 151936

_decode = torch.ops.qwen_megakernel_C.decode


def _load_weights_tts(model_name: str, verbose: bool) -> tuple[dict, object, float]:
    """Load TTS model via qwen_tts, extract talker decoder state, and return (state_dict, tokenizer).
    State dict uses megakernel key names: model.embed_tokens.weight, model.layers.*, model.norm.weight, lm_head.weight.
    """
    from qwen_tts import Qwen3TTSModel  # type: ignore[import-untyped]

    if verbose:
        print(f"Loading TTS model (talker decoder): {model_name}...")
    tts_model = Qwen3TTSModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    # Qwen3TTSModel wraps .model (Qwen3TTSForConditionalGeneration) which has .talker
    talker = getattr(tts_model, "talker", None) or getattr(tts_model.model, "talker", None)
    if talker is None:
        raise AttributeError(
            "TTS model has no 'talker' (expected on model.talker for Qwen3TTSForConditionalGeneration)"
        )
    # Talker state_dict: keys like model.codec_embedding.weight, model.layers.*, model.norm.weight, codec_head.weight
    raw = talker.state_dict()
    state = {}
    for k, v in raw.items():
        if not (k.startswith("model.") or k.startswith("codec_head.")):
            continue  # skip code_predictor, text_projection, etc.
        if k == "model.codec_embedding.weight":
            state["model.embed_tokens.weight"] = v.contiguous()
        elif k == "codec_head.weight":
            state["lm_head.weight"] = v.contiguous()
        elif k.startswith("model."):
            state[k] = v.contiguous() if hasattr(v, "contiguous") else v
    # Tokenizer: TTS repo may not expose AutoTokenizer; try same repo then fallback to Qwen3-0.6B
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        if verbose:
            print("   Using Qwen/Qwen3-0.6B tokenizer (TTS repo tokenizer not available).")
    # RoPE: talker config may have rope_theta=1000000 (rope_scaling); our kernel uses 1D RoPE with default 10000.
    rope_theta = float(getattr(talker.config, "rope_theta", 10000.0))
    if rope_theta > 100000:
        rope_theta = 10000.0
        if verbose:
            print("RoPE theta: 10000.0 (using default for 1D kernel; config had scaling value)")
    elif verbose:
        print(f"RoPE theta: {rope_theta}")
    del tts_model
    torch.cuda.empty_cache()
    return state, tokenizer, rope_theta


def load_weights(model_name="Qwen/Qwen3-0.6B", verbose: bool = True):
    """Load Qwen3-0.6B or Qwen3-TTS weights from HuggingFace into GPU tensors."""
    if not verbose:
        import os

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    if not verbose:
        hf_logging.set_verbosity_error()
        try:
            hf_logging.disable_progress_bar()
        except AttributeError:
            pass
        try:
            from huggingface_hub import logging as hf_hub_logging

            hf_hub_logging.set_verbosity_error()
        except Exception:
            pass

    is_tts = "tts" in model_name.lower() or "TTS" in model_name
    if is_tts:
        try:
            state, tokenizer, rope_theta = _load_weights_tts(model_name, verbose)
        except ImportError as e:
            if verbose:
                print("⚠️  qwen-tts required for TTS models. Install with: pip install qwen-tts")
            raise e
    else:
        if verbose:
            print(f"Loading {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        state = model.state_dict()
        # Resolve effective RoPE theta: Qwen3 uses rope_scaling.rope_theta (e.g. 1000000), not config.rope_theta (10000)
        rope_theta = float(getattr(model.config, "rope_theta", 10000.0))
        rope_scaling = getattr(model.config, "rope_scaling", None)
        if isinstance(rope_scaling, dict) and "rope_theta" in rope_scaling:
            rope_theta = float(rope_scaling["rope_theta"])
        if verbose:
            print(f"RoPE theta: {rope_theta}")
        del model
        torch.cuda.empty_cache()

    # RoPE tables (rope_theta already set above from config or TTS talker config) — computed in f32, stored as bf16 on GPU.
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()

    # Per-layer weight list (11 tensors per layer, flattened)
    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}."
        layer_weights.extend(
            [
                state[p + "input_layernorm.weight"].contiguous(),
                state[p + "self_attn.q_proj.weight"].contiguous(),
                state[p + "self_attn.k_proj.weight"].contiguous(),
                state[p + "self_attn.v_proj.weight"].contiguous(),
                state[p + "self_attn.q_norm.weight"].contiguous(),
                state[p + "self_attn.k_norm.weight"].contiguous(),
                state[p + "self_attn.o_proj.weight"].contiguous(),
                state[p + "post_attention_layernorm.weight"].contiguous(),
                state[p + "mlp.gate_proj.weight"].contiguous(),
                state[p + "mlp.up_proj.weight"].contiguous(),
                state[p + "mlp.down_proj.weight"].contiguous(),
            ]
        )

    embed_weight = state["model.embed_tokens.weight"].contiguous()

    # Explicit lm_head.weight load with shape guard.
    # Never alias embed_weight when the separate tensor exists: transformers 5.x
    # stores both as distinct allocations even when tie_word_embeddings=True.
    # For TTS variants with a different vocab head this will load the correct tensor.
    if "lm_head.weight" in state:
        lm_head_weight = state["lm_head.weight"].contiguous()
    else:
        lm_head_weight = embed_weight

    if is_tts:
        # TTS talker codec_head may use a different vocab size (e.g. 3072 for 12Hz); only require hidden size.
        assert lm_head_weight.shape[1] == HIDDEN_SIZE, (
            f"lm_head.weight shape {tuple(lm_head_weight.shape)} has hidden_size {lm_head_weight.shape[1]} != {HIDDEN_SIZE}."
        )
    else:
        assert lm_head_weight.shape == (VOCAB_SIZE, HIDDEN_SIZE), (
            f"lm_head.weight shape {tuple(lm_head_weight.shape)} != "
            f"expected ({VOCAB_SIZE}, {HIDDEN_SIZE}). "
            "Update VOCAB_SIZE / HIDDEN_SIZE constants if using a different model."
        )

    weights = dict(
        embed_weight=embed_weight,
        layer_weights=layer_weights,
        final_norm_weight=state["model.norm.weight"].contiguous(),
        lm_head_weight=lm_head_weight,
        cos_table=cos_table,
        sin_table=sin_table,
        rope_theta_used=rope_theta,
    )

    torch.cuda.empty_cache()
    return weights, tokenizer


def _pack_layer_weights(layer_weights: list[torch.Tensor]) -> torch.Tensor:
    """Pack 11-tensor-per-layer flat list into a device blob of LDGLayerWeights structs."""
    ptr_size = 8  # 64-bit pointers
    n_ptrs = 11
    struct_bytes = n_ptrs * ptr_size
    buf = bytearray(NUM_LAYERS * struct_bytes)
    for i in range(NUM_LAYERS):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    t = torch.frombuffer(buf, dtype=torch.uint8).cuda()
    return t


class Decoder:
    """Stateful decoder wrapping the Qwen Megakernel torch ops."""

    def __init__(
        self,
        weights=None,
        tokenizer=None,
        model_name="Qwen/Qwen3-0.6B",
        verbose: bool = True,
    ):
        if weights is None:
            weights, tokenizer = load_weights(model_name, verbose=verbose)
        self.tokenizer = tokenizer
        self._position = 0
        self._rope_theta_used = weights.get("rope_theta_used")

        # Keep references so tensors stay alive (prevents GC of weight memory).
        self._weights = weights

        # Model weights (read-only, shared across calls)
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._cos_table = weights["cos_table"]
        self._sin_table = weights["sin_table"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])

        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)

        # KV cache
        self._k_cache = torch.zeros(
            NUM_LAYERS,
            NUM_KV_HEADS,
            MAX_SEQ_LEN,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        # Scratch buffers (single-token decode)
        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        self._act = torch.empty(HIDDEN_SIZE, **f32)
        self._res = torch.empty(HIDDEN_SIZE, **f32)
        self._q = torch.empty(Q_SIZE, **f32)
        self._k = torch.empty(KV_SIZE, **f32)
        self._v = torch.empty(KV_SIZE, **f32)
        self._attn_out = torch.empty(Q_SIZE, **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out = torch.empty(HIDDEN_SIZE, **f32)
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")
        self._logits_buffer = torch.empty(VOCAB_SIZE, dtype=torch.float32, device="cuda")

    def step(
        self,
        token_id: int,
        rope_position_override: Optional[int] = None,
        return_logits: bool = False,
    ) -> Union[int, Tuple[int, torch.Tensor]]:
        """Decode one token. Returns the next token id, or (token_id, logits) when return_logits=True."""
        rop = -1 if rope_position_override is None else int(rope_position_override)
        logits_out = self._logits_buffer if return_logits else None
        _decode(
            self._out_token,
            token_id,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
            rop,
            logits_out,
        )
        self._position += 1
        if return_logits:
            return self._out_token.item(), self._logits_buffer.clone()
        return self._out_token.item()

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    @property
    def position(self) -> int:
        return self._position

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        self.reset()
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        for tid in ids[:-1]:
            self.step(tid)
        _gen = torch.ops.qwen_megakernel_C.generate_nosync
        output_ids = _gen(
            ids[-1],
            max_tokens,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += max_tokens
        out = output_ids.cpu().tolist()
        eos = self.tokenizer.eos_token_id
        if eos in out:
            out = out[: out.index(eos)]
        return self.tokenizer.decode(out, skip_special_tokens=True)


def generate(prompt: str, max_tokens: int = 100, verbose: bool = True) -> str:
    """One-shot convenience: load model, generate, return text."""
    return Decoder(verbose=verbose).generate(prompt, max_tokens)
