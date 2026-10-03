"""CuTe DSL (SM100/SM103) backend for gather attention.

Calls FlashAttention-4's sparse MLA forward and backward with ``gather_kv_indices``
for index-gather mode, behind one registered forward/backward operator pair so
``torch.compile`` and fake-tensor tracers (``make_fx``) see an opaque op instead of
FA4's Python launcher. FA4 owns the kernels, compilation caching and workspaces.

Constraints
-----------
- head_dim = 512, 1 <= nheads <= 128, share_kv = True (fewer than 128 heads are
  zero-padded to FA4's 64/128-head tiles in-kernel via TMA out-of-bounds)
- dtype = bfloat16, SM100 or SM103 (compute capability 10.0 or 10.3)
- Requires FA4 4.0.0b32+ for sparse MLA attention sinks and, with fewer than 128
  heads, sparse-MLA head padding; 4.0.0b33+ to recompute probabilities in backward
  for any head count

TODO: the operators call FA4's private ``_flash_attn_fwd`` and
``_flash_attn_bwd_sparse_mla``, and the fakes restate their allocations. Re-run
test_gather_attn_cute.py and test_gather_attn_varlen.py on any flash-attn-4 bump.
"""

from __future__ import annotations

import inspect
import warnings
from functools import cache

import torch

from . import cute_ops  # noqa: F401  (operator schemas)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))


def _constraint_violation(query: torch.Tensor, share_kv: bool) -> Exception | None:
    """Return the error for metadata this backend cannot run, or None when it qualifies."""
    _b, h, _s, d = query.shape
    if query.device.type != "cuda":
        return ValueError("CuTe backend requires CUDA tensors.")
    if torch.cuda.get_device_capability(query.device) not in SUPPORTED_CAPABILITIES:
        return ValueError("CuTe backend requires SM100 or SM103.")
    if query.dtype != torch.bfloat16:
        return TypeError("CuTe backend requires bfloat16.")
    if not share_kv:
        return ValueError("CuTe backend requires share_kv=True.")
    if d != 512:
        return ValueError(f"CuTe backend requires head_dim=512, got {d}.")
    if not 0 < h <= 128:
        return ValueError(f"CuTe backend requires at most 128 query heads, got {h}.")
    return None


# Dynamo cannot trace FA4's import probe; its result is fixed per process.
@torch.compiler.assume_constant_result
@cache
def _fa4_available(with_sink: bool, *, padded_heads: bool = False) -> bool:
    """Probe the optional dependency once, only after tensor metadata qualifies."""
    try:
        from flash_attn.cute.interface import flash_attn_func  # noqa: F401

        if padded_heads:
            # Added with FA4's arbitrary-head sparse MLA kernels; older FA4 requires H=128.
            from flash_attn.cute.pack_gqa import sparse_mla_qhead_tile  # noqa: F401

        if with_sink:
            from flash_attn.cute.flash_fwd_mla_sm100 import FlashAttentionMLAForwardSm100

            return (
                "learnable_sink"
                in inspect.signature(FlashAttentionMLAForwardSm100.__call__).parameters
            )
    except ImportError:
        return False
    return True


def is_supported(query, attention_sink, share_kv) -> bool:
    """Check metadata and optional FA4 features without launching a kernel."""
    if _constraint_violation(query, share_kv) is not None:
        return False
    # FA4 requires a unit leading stride. Its contiguous repair skips singleton views,
    # which PyTorch considers contiguous even when their stride is not one.
    if (
        attention_sink is not None
        and attention_sink.numel() == 1
        and attention_sink.stride(0) != 1
    ):
        return False
    return _fa4_available(attention_sink is not None, padded_heads=query.shape[1] != 128)


def _check_backward_mode() -> None:
    if torch.are_deterministic_algorithms_enabled():
        message = (
            "CuTe gather attention does not support deterministic backward; "
            "use kernel_options={'backend': 'triton'} for the forward call."
        )
        if torch.is_deterministic_algorithms_warn_only_enabled():
            warnings.warn(message, stacklevel=2)
        else:
            raise RuntimeError(message)


# ---------------------------------------------------------------------------
# Opaque operators
# ---------------------------------------------------------------------------

# The schemas are defined in cute_ops.py, which the package imports eagerly.


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


def _fa4_inputs(
    query: torch.Tensor,
    local_kv: torch.Tensor,
    sparse_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    sliding_window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """FA4 sparse-MLA inputs: (qv, shared kv, gather indices, varlen offsets).

    Passing k=v (the same tensor) with hdim=512 selects FA4's sparse MLA path.
    """
    from .indices import build_gather_indices
    from .packed_kv import pack_kv

    gather_indices = build_gather_indices(
        kv_indices,
        cu_seqlens,
        cu_seqlens_k,
        sliding_window_size,
        sparse_kv_len=sparse_kv.shape[2],
    )
    if cu_seqlens is None:
        kv = torch.cat([local_kv, sparse_kv], dim=2).permute(0, 2, 1, 3)
        return _to_fa4_layout(query, False), kv, gather_indices, {}
    kv, cu_q, cu_kv = pack_kv(local_kv, sparse_kv, cu_seqlens, cu_seqlens_k)
    offsets = {"cu_seqlens_q": cu_q, "cu_seqlens_k": cu_kv}
    return _to_fa4_layout(query, True), kv, gather_indices[0], offsets


def _gather_attn_cute_fwd_impl(
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
    from flash_attn.cute.interface import _flash_attn_fwd

    qv, kv, gather_indices, offsets = _fa4_inputs(
        query, local_kv, sparse_kv, kv_indices, cu_seqlens, cu_seqlens_k, sliding_window_size
    )
    # FA4 sizes its backward buffers (o_lo, and p/row_max unless recomputing) from its
    # inputs' requires_grad, which tracing does not preserve: decide from needs_backward.
    qv = qv.detach().requires_grad_(needs_backward)
    out, lse, p, row_max, o_lo = _flash_attn_fwd(
        None,
        None,
        kv.detach(),
        qv=qv,
        softmax_scale=scale,
        causal=False,
        learnable_sink=None if attention_sink is None else attention_sink.detach(),
        pack_gqa=True,
        return_lse=True,
        gather_kv_indices=gather_indices,
        gather_bwd_recompute_p=bwd_recompute_p,
        **offsets,
    )
    packed = cu_seqlens is not None
    # Fixed arity: buffers FA4 did not allocate become distinct empty outputs.
    return (
        _from_fa4_layout(out, packed),
        _from_fa4_layout(lse, packed),
        query.new_empty((0,)) if p is None else p,
        query.new_empty((0,), dtype=torch.float32) if row_max is None else row_max,
        query.new_empty((0,)) if o_lo is None else o_lo,
    )


def _split_dense_grad_kv(
    grad_kv: torch.Tensor, local_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split FA4's dense (batch, T + S, 1, D) KV gradient into local and sparse gradients."""
    grad_local, grad_sparse = grad_kv.permute(0, 2, 1, 3).split(
        [local_len, grad_kv.shape[1] - local_len], dim=2
    )
    # Operator outputs may not alias each other: give the sparse gradient its own storage.
    return grad_local, grad_sparse.clone()


def _gather_attn_cute_bwd_impl(
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
    from flash_attn.cute.interface import _flash_attn_bwd_sparse_mla

    from .packed_kv import _launch_pack_kv

    # Determinism can be enabled after forward; honor its strict/warn-only setting.
    _check_backward_mode()
    # Rebuild the indices and KV pool rather than saving them: both are cheap next to
    # the attention backward.
    qv, kv, gather_indices, offsets = _fa4_inputs(
        query, local_kv, sparse_kv, kv_indices, cu_seqlens, cu_seqlens_k, sliding_window_size
    )
    packed = cu_seqlens is not None
    # Preallocated in the query's layout so the fake can state dQ's strides.
    grad_query = torch.empty_like(query)
    _, _, grad_kv, _, grad_sink = _flash_attn_bwd_sparse_mla(
        None,
        None,
        kv,
        qv,
        _to_fa4_layout(output, packed),
        _to_fa4_layout(grad_output, packed),
        _to_fa4_layout(lse, packed),
        None if bwd_recompute_p else p,
        None if bwd_recompute_p else row_max,
        gather_indices,
        learnable_sink=attention_sink,
        softmax_scale=scale,
        causal=False,
        dqv=_to_fa4_layout(grad_query, packed),
        recompute_p=bwd_recompute_p,
        o_lo=o_lo,
        **offsets,
    )
    # FA4 accumulates dKV in FP32; cast once before splitting it into the two inputs.
    grad_kv = grad_kv.to(local_kv.dtype)
    if packed:
        grad_local = local_kv.new_empty(local_kv.shape)
        grad_sparse = sparse_kv.new_empty(sparse_kv.shape)
        _launch_pack_kv(
            grad_local,
            grad_sparse,
            grad_kv,
            cu_seqlens,
            cu_seqlens_k,
            None,
            None,
            pack=False,
        )
    else:
        grad_local, grad_sparse = _split_dense_grad_kv(grad_kv, local_kv.shape[2])
    if grad_sink is None:
        grad_sink = query.new_empty((0,), dtype=torch.float32)
    return grad_query, grad_local, grad_sparse, grad_sink


torch.library.impl("attn_gym::_gather_attn_cute_fwd", "CUDA", _gather_attn_cute_fwd_impl)
torch.library.impl("attn_gym::_gather_attn_cute_bwd", "CUDA", _gather_attn_cute_bwd_impl)


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


_gather_attn_cute_fwd_op = torch.ops.attn_gym._gather_attn_cute_fwd.default
_gather_attn_cute_bwd_op = torch.ops.attn_gym._gather_attn_cute_bwd.default


class _GatherAttnCuteFunction(torch.autograd.Function):
    """Autograd wrapper around the opaque CuTe operators."""

    @staticmethod
    def forward(
        ctx,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, lse, p, row_max, o_lo = _gather_attn_cute_fwd_op(
            query,
            local_kv,
            sparse_kv,
            kv_indices,
            attention_sink,
            cu_seqlens,
            cu_seqlens_k,
            sliding_window_size,
            scale,
            bwd_recompute_p,
            needs_backward,
        )
        ctx.save_for_backward(
            query,
            local_kv,
            sparse_kv,
            kv_indices,
            attention_sink,
            cu_seqlens,
            cu_seqlens_k,
            output,
            lse,
            p,
            row_max,
            o_lo,
        )
        ctx.sliding_window_size = sliding_window_size
        ctx.scale = scale
        ctx.bwd_recompute_p = bwd_recompute_p
        # Sparse MLA ignores dLSE. Match Triton's nondifferentiable auxiliary contract.
        ctx.mark_non_differentiable(lse)
        return output, lse

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor, grad_lse: torch.Tensor | None):
        saved = ctx.saved_tensors
        attention_sink = saved[4]
        grad_query, grad_local, grad_sparse, grad_sink = _gather_attn_cute_bwd_op(
            *saved,
            grad_output,
            ctx.sliding_window_size,
            ctx.scale,
            ctx.bwd_recompute_p,
        )
        return (
            grad_query,
            grad_local,
            grad_sparse,
            None,
            None if attention_sink is None else grad_sink,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def gather_attn(
    query: torch.Tensor,
    local_kv: torch.Tensor,
    sparse_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    attention_sink: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    sliding_window_size: int,
    share_kv: bool = True,
    *,
    scale: float,
    bwd_recompute_p: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CuTe DSL (SM100/SM103) forward+backward for gather attention.

    Optional per-head attention sinks are forwarded to FA4, which owns their gradients.
    ``bwd_recompute_p`` recomputes the attention probabilities in backward instead of
    saving them in forward. Whether to allocate backward buffers is decided at forward
    time, from grad mode and the inputs' ``requires_grad``.

    Returns:
        Tuple of (output, lse) where output has shape (batch, heads, seq, head_dim)
        and lse has shape (batch, heads, seq).
    """
    if (error := _constraint_violation(query, share_kv)) is not None:
        raise error
    needs_backward = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (query, local_kv, sparse_kv, attention_sink)
    )
    return _GatherAttnCuteFunction.apply(
        query,
        local_kv,
        sparse_kv,
        kv_indices,
        attention_sink,
        cu_seqlens,
        cu_seqlens_k,
        sliding_window_size,
        scale,
        bwd_recompute_p,
        needs_backward,
    )
