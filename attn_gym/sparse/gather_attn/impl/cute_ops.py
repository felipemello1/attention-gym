"""Schemas of the CuTe gather-attention operators.

The package imports this module so callers can name the operators (e.g. in an
activation-checkpointing save list) before the first call. ``cute.py`` registers their
CUDA kernels and fakes; it loads on the first call because it imports ``torch._dynamo``.
"""

import torch

torch.library.define(
    "attn_gym::_gather_attn_cute_fwd",
    "(Tensor query, Tensor local_kv, Tensor sparse_kv, Tensor kv_indices, "
    "Tensor? attention_sink, Tensor? cu_seqlens, Tensor? cu_seqlens_k, "
    "int sliding_window_size, float scale, bool bwd_recompute_p, bool needs_backward) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::_gather_attn_cute_bwd",
    "(Tensor query, Tensor local_kv, Tensor sparse_kv, Tensor kv_indices, "
    "Tensor? attention_sink, Tensor? cu_seqlens, Tensor? cu_seqlens_k, Tensor output, "
    "Tensor lse, Tensor p, Tensor row_max, Tensor o_lo, Tensor grad_output, "
    "int sliding_window_size, float scale, bool bwd_recompute_p) "
    "-> (Tensor, Tensor, Tensor, Tensor)",
)
