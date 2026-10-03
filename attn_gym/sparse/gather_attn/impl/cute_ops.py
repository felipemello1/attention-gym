"""Schemas, fakes and CUDA entry points of the CuTe gather-attention operators.

The package imports this module, so the operators are complete before the first
``gather_attn`` call: callers can name them (e.g. in an activation-checkpointing save
list) and run graphs that contain them (e.g. a deserialized traced graph). The kernels
live in ``cute.py``, whose module-level ``torch.compiler.assume_constant_result`` loads
``torch._dynamo``; the CUDA entry points import it on their first call.
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


def _to_fa4_layout(tensor: torch.Tensor, packed: bool) -> torch.Tensor:
    """(batch, heads, seq, ...) -> FA4 (batch, seq, heads, ...) or packed (tokens, heads, ...)."""
    if packed:
        return tensor.squeeze(0).transpose(0, 1)
    return tensor.transpose(1, 2)


def _from_fa4_layout(tensor: torch.Tensor, packed: bool) -> torch.Tensor:
    """Inverse of _to_fa4_layout."""
    if packed:
        return tensor.unsqueeze(0).transpose(1, 2)
    return tensor.transpose(1, 2)


def _split_dense_grad_kv(
    grad_kv: torch.Tensor, local_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split FA4's dense (batch, T + S, 1, D) KV gradient into local and sparse gradients."""
    grad_local, grad_sparse = grad_kv.permute(0, 2, 1, 3).split(
        [local_len, grad_kv.shape[1] - local_len], dim=2
    )
    # Operator outputs may not alias each other: give the sparse gradient its own storage.
    return grad_local, grad_sparse.clone()


# The CUDA kernels live in cute.py; import it on the first call, not at package import.


def _gather_attn_cute_fwd_cuda(*args):
    from .cute import _gather_attn_cute_fwd_impl

    return _gather_attn_cute_fwd_impl(*args)


def _gather_attn_cute_bwd_cuda(*args):
    from .cute import _gather_attn_cute_bwd_impl

    return _gather_attn_cute_bwd_impl(*args)


torch.library.impl("attn_gym::_gather_attn_cute_fwd", "CUDA", _gather_attn_cute_fwd_cuda)
torch.library.impl("attn_gym::_gather_attn_cute_bwd", "CUDA", _gather_attn_cute_bwd_cuda)


# FA4 compiles its kernels from real data pointers, which compile-time fake tensors refuse,
# so the fakes restate the allocations of FA4's sparse-MLA launchers.


@torch.library.register_fake("attn_gym::_gather_attn_cute_fwd")
def _gather_attn_cute_fwd_fake(
    query: torch.Tensor,
    local_kv: torch.Tensor,
    sparse_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    attention_sink: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    sliding_window_size: int,
    scale: float,
    bwd_recompute_p: bool,
    needs_backward: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, tokens, head_dim = query.shape
    packed = cu_seqlens is not None
    rows = (tokens,) if packed else (batch, tokens)
    # Must match build_gather_indices' slot padding (indices.py).
    slots = -(-max(sliding_window_size + kv_indices.shape[2], 1) // 128) * 128
    # FA4 returns before allocating backward buffers when there are no query tokens; the
    # public API already rejects that shape.
    needs_backward = needs_backward and tokens > 0
    saves_p = needs_backward and not bwd_recompute_p
    return (
        _from_fa4_layout(query.new_empty((*rows, heads, head_dim)), packed),
        _from_fa4_layout(query.new_empty((*rows, heads), dtype=torch.float32), packed),
        query.new_empty((*rows, heads, slots) if saves_p else (0,)),
        query.new_empty((*rows, slots // 128, heads) if saves_p else (0,), dtype=torch.float32),
        query.new_empty((*rows, heads, head_dim) if needs_backward else (0,)),
    )


@torch.library.register_fake("attn_gym::_gather_attn_cute_bwd")
def _gather_attn_cute_bwd_fake(
    query: torch.Tensor,
    local_kv: torch.Tensor,
    sparse_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    attention_sink: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    output: torch.Tensor,
    lse: torch.Tensor,
    p: torch.Tensor,
    row_max: torch.Tensor,
    o_lo: torch.Tensor,
    grad_output: torch.Tensor,
    sliding_window_size: int,
    scale: float,
    bwd_recompute_p: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if cu_seqlens is None:
        # FA4's dKV copies the layout of the concatenated KV it reads.
        kv = torch.cat([local_kv, sparse_kv], dim=2).permute(0, 2, 1, 3)
        grad_local, grad_sparse = _split_dense_grad_kv(torch.empty_like(kv), local_kv.shape[2])
    else:
        grad_local = local_kv.new_empty(local_kv.shape)
        grad_sparse = sparse_kv.new_empty(sparse_kv.shape)
    grad_sink = (
        query.new_empty((0,), dtype=torch.float32)
        if attention_sink is None
        else torch.empty_like(attention_sink, memory_format=torch.contiguous_format)
    )
    return torch.empty_like(query), grad_local, grad_sparse, grad_sink
