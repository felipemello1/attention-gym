# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused GDN input preparation: q/k L2 norm, softplus decay gate, and sigmoid write gate.

    q_out = q / sqrt(sum_d q^2 + eps)                      [1, T, HK, D]
    k_out = k / sqrt(sum_d k^2 + eps)                      [1, T, HK, D]
    gate  = -exp(A_log) * softplus(raw_gate + dt_bias)     [1, T, H], FP32
    beta  = sigmoid(raw_beta)                              [1, T, H]

Each direction is one launch over the grid ``(token blocks, 2 * HK + 1)``: slots ``[0, HK)``
normalize q heads, ``[HK, 2 * HK)`` normalize k heads, and the last slot computes both gates for
all ``H`` heads. The backward recomputes ``y`` in FP32 from the exact input and the saved norms.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from attn_gym.linear._delta_rule.triton.softplus_gate import _sigmoid, _softplus

BLOCK_TOKENS = 32


@triton.jit
def _l2norm_tile_fwd(
    x, y, rstd, o_t, m_t, i_h, X_ROW_STRIDE, HK: tl.constexpr, D: tl.constexpr, eps
):
    o_d = tl.arange(0, D)
    b_x = tl.load(
        x + o_t[:, None] * X_ROW_STRIDE + i_h * D + o_d[None, :], mask=m_t[:, None], other=0.0
    ).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    o_y = (o_t[:, None] * HK + i_h) * D + o_d[None, :]
    tl.store(y + o_y, (b_x * b_rstd[:, None]).to(y.dtype.element_ty), mask=m_t[:, None])
    tl.store(rstd + o_t * 2 * HK + i_h, b_rstd, mask=m_t)


@triton.jit
def _l2norm_tile_bwd(
    x, rstd, dy, dx, o_t, m_t, i_h, X_ROW_STRIDE, HK: tl.constexpr, D: tl.constexpr
):
    o_d = tl.arange(0, D)
    b_x = tl.load(
        x + o_t[:, None] * X_ROW_STRIDE + i_h * D + o_d[None, :], mask=m_t[:, None], other=0.0
    ).to(tl.float32)
    o_y = (o_t[:, None] * HK + i_h) * D + o_d[None, :]
    b_dy = tl.load(dy + o_y, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_rstd = tl.load(rstd + o_t * 2 * HK + i_h, mask=m_t, other=0.0)
    # dx = rstd * (dy - y * <dy, y>), with y recomputed in FP32 from the exact input.
    b_y = b_x * b_rstd[:, None]
    b_dx = b_rstd[:, None] * (b_dy - b_y * tl.sum(b_dy * b_y, 1)[:, None])
    tl.store(dx + o_y, b_dx.to(dx.dtype.element_ty), mask=m_t[:, None])


@triton.jit(do_not_specialize=["T"])
def gdn_prepare_fwd_kernel(
    q,
    k,
    raw_gate,
    raw_beta,
    A_log,
    dt_bias,
    q_out,
    k_out,
    gate,
    beta,
    rstd,
    eps,
    T,
    Q_ROW_STRIDE,
    K_ROW_STRIDE,
    G_ROW_STRIDE,
    B_ROW_STRIDE,
    HK: tl.constexpr,
    H: tl.constexpr,
    BH: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    FASTMATH: tl.constexpr,
):
    o_t = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    m_t = o_t < T
    i_slot = tl.program_id(1)
    if i_slot < HK:
        _l2norm_tile_fwd(q, q_out, rstd, o_t, m_t, i_slot, Q_ROW_STRIDE, HK, D, eps)
    elif i_slot < 2 * HK:
        _l2norm_tile_fwd(k, k_out, rstd + HK, o_t, m_t, i_slot - HK, K_ROW_STRIDE, HK, D, eps)
    else:
        o_h = tl.arange(0, BH)
        m_h = o_h < H
        mask = m_t[:, None] & m_h[None, :]
        b_a = tl.load(raw_gate + o_t[:, None] * G_ROW_STRIDE + o_h[None, :], mask=mask, other=0.0)
        b_b = tl.load(raw_beta + o_t[:, None] * B_ROW_STRIDE + o_h[None, :], mask=mask, other=0.0)
        s = b_a.to(tl.float32) + tl.load(dt_bias + o_h, mask=m_h, other=0.0)[None, :]
        b_A_log = tl.load(A_log + o_h, mask=m_h, other=0.0)[None, :]
        amplitude = tl.exp(b_A_log) if FASTMATH else libdevice.exp(b_A_log)
        b_beta = _sigmoid(b_b.to(tl.float32), FASTMATH)
        o_g = o_t[:, None] * H + o_h[None, :]
        tl.store(gate + o_g, -amplitude * _softplus(s, FASTMATH), mask=mask)
        tl.store(beta + o_g, b_beta.to(beta.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["T"])
def gdn_prepare_bwd_kernel(
    q,
    k,
    raw_gate,
    raw_beta,
    A_log,
    dt_bias,
    rstd,
    d_q_out,
    d_k_out,
    d_gate,
    d_beta,
    d_q,
    d_k,
    d_raw_gate,
    d_raw_beta,
    d_A_log_partial,
    d_dt_bias_partial,
    T,
    Q_ROW_STRIDE,
    K_ROW_STRIDE,
    G_ROW_STRIDE,
    B_ROW_STRIDE,
    HK: tl.constexpr,
    H: tl.constexpr,
    BH: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    FASTMATH: tl.constexpr,
):
    i_block = tl.program_id(0)
    o_t = i_block.to(tl.int64) * BT + tl.arange(0, BT)
    m_t = o_t < T
    i_slot = tl.program_id(1)
    if i_slot < HK:
        _l2norm_tile_bwd(q, rstd, d_q_out, d_q, o_t, m_t, i_slot, Q_ROW_STRIDE, HK, D)
    elif i_slot < 2 * HK:
        _l2norm_tile_bwd(k, rstd + HK, d_k_out, d_k, o_t, m_t, i_slot - HK, K_ROW_STRIDE, HK, D)
    else:
        o_h = tl.arange(0, BH)
        m_h = o_h < H
        mask = m_t[:, None] & m_h[None, :]
        b_a = tl.load(raw_gate + o_t[:, None] * G_ROW_STRIDE + o_h[None, :], mask=mask, other=0.0)
        b_b = tl.load(raw_beta + o_t[:, None] * B_ROW_STRIDE + o_h[None, :], mask=mask, other=0.0)
        s = b_a.to(tl.float32) + tl.load(dt_bias + o_h, mask=m_h, other=0.0)[None, :]
        b_A_log = tl.load(A_log + o_h, mask=m_h, other=0.0)[None, :]
        amplitude = tl.exp(b_A_log) if FASTMATH else libdevice.exp(b_A_log)
        b_beta = _sigmoid(b_b.to(tl.float32), FASTMATH)
        o_g = o_t[:, None] * H + o_h[None, :]
        b_dgate = tl.load(d_gate + o_g, mask=mask, other=0.0)
        b_dbeta = tl.load(d_beta + o_g, mask=mask, other=0.0).to(tl.float32)
        # gate = -amplitude * softplus(s): d/ds = -amplitude * sigmoid(s); d/dA_log = gate.
        d_s = -amplitude * _sigmoid(s, FASTMATH) * b_dgate
        d_A_log = -amplitude * _softplus(s, FASTMATH) * b_dgate
        d_b = b_dbeta * b_beta * (1.0 - b_beta)
        tl.store(d_raw_gate + o_g, d_s.to(d_raw_gate.dtype.element_ty), mask=mask)
        tl.store(d_raw_beta + o_g, d_b.to(d_raw_beta.dtype.element_ty), mask=mask)
        # Masked tokens loaded d_gate == 0, so they add exactly zero to the block partials.
        tl.store(d_A_log_partial + i_block * H + o_h, tl.sum(d_A_log, 0), mask=m_h)
        tl.store(d_dt_bias_partial + i_block * H + o_h, tl.sum(d_s, 0), mask=m_h)


torch.library.define(
    "attn_gym::gdn_prepare_fwd",
    "(Tensor q, Tensor k, Tensor raw_gate, Tensor raw_beta, Tensor A_log, Tensor dt_bias, "
    "float eps, bool fastmath) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::gdn_prepare_bwd",
    "(Tensor q, Tensor k, Tensor raw_gate, Tensor raw_beta, Tensor A_log, Tensor dt_bias, "
    "Tensor rstd, Tensor d_q_out, Tensor d_k_out, Tensor d_gate, Tensor d_beta, bool fastmath) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
)


@torch.library.impl("attn_gym::gdn_prepare_fwd", "CUDA")
def _gdn_prepare_fwd_cuda(q, k, raw_gate, raw_beta, A_log, dt_bias, eps, fastmath):
    _, tokens, key_heads, head_dim = q.shape
    heads = raw_gate.shape[2]
    q_out = torch.empty_like(q, memory_format=torch.contiguous_format)
    k_out = torch.empty_like(k, memory_format=torch.contiguous_format)
    gate = torch.empty_like(raw_gate, dtype=torch.float32, memory_format=torch.contiguous_format)
    beta = torch.empty_like(raw_beta, memory_format=torch.contiguous_format)
    # FP32 inverse norms of the q heads, then the k heads; the backward reuses them.
    rstd = q.new_empty(1, tokens, 2 * key_heads, dtype=torch.float32)
    gdn_prepare_fwd_kernel[(triton.cdiv(tokens, BLOCK_TOKENS), 2 * key_heads + 1)](
        q,
        k,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        q_out,
        k_out,
        gate,
        beta,
        rstd,
        eps,
        tokens,
        q.stride(1),
        k.stride(1),
        raw_gate.stride(1),
        raw_beta.stride(1),
        HK=key_heads,
        H=heads,
        BH=triton.next_power_of_2(heads),
        D=head_dim,
        BT=BLOCK_TOKENS,
        FASTMATH=fastmath,
        num_warps=4,
        enable_reflect_ftz=fastmath,
    )
    return q_out, k_out, gate, beta, rstd


@torch.library.impl("attn_gym::gdn_prepare_bwd", "CUDA")
def _gdn_prepare_bwd_cuda(
    q, k, raw_gate, raw_beta, A_log, dt_bias, rstd, d_q_out, d_k_out, d_gate, d_beta, fastmath
):
    _, tokens, key_heads, head_dim = q.shape
    heads = raw_gate.shape[2]
    blocks = triton.cdiv(tokens, BLOCK_TOKENS)
    d_q, d_k, d_raw_gate, d_raw_beta = (
        torch.empty_like(t, memory_format=torch.contiguous_format)
        for t in (q, k, raw_gate, raw_beta)
    )
    d_A_log_partial = torch.empty(blocks, heads, dtype=torch.float32, device=q.device)
    d_dt_bias_partial = torch.empty_like(d_A_log_partial)
    gdn_prepare_bwd_kernel[(blocks, 2 * key_heads + 1)](
        q,
        k,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        rstd,
        d_q_out.contiguous(),
        d_k_out.contiguous(),
        d_gate.contiguous(),
        d_beta.contiguous(),
        d_q,
        d_k,
        d_raw_gate,
        d_raw_beta,
        d_A_log_partial,
        d_dt_bias_partial,
        tokens,
        q.stride(1),
        k.stride(1),
        raw_gate.stride(1),
        raw_beta.stride(1),
        HK=key_heads,
        H=heads,
        BH=triton.next_power_of_2(heads),
        D=head_dim,
        BT=BLOCK_TOKENS,
        FASTMATH=fastmath,
        num_warps=4,
        enable_reflect_ftz=fastmath,
    )
    return d_q, d_k, d_raw_gate, d_raw_beta, d_A_log_partial.sum(0), d_dt_bias_partial.sum(0)


@torch.library.register_fake("attn_gym::gdn_prepare_fwd")
def _gdn_prepare_fwd_fake(q, k, raw_gate, raw_beta, A_log, dt_bias, eps, fastmath):
    del A_log, dt_bias, eps, fastmath
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(k, memory_format=torch.contiguous_format),
        torch.empty_like(raw_gate, dtype=torch.float32, memory_format=torch.contiguous_format),
        torch.empty_like(raw_beta, memory_format=torch.contiguous_format),
        q.new_empty(1, q.shape[1], 2 * q.shape[2], dtype=torch.float32),
    )


@torch.library.register_fake("attn_gym::gdn_prepare_bwd")
def _gdn_prepare_bwd_fake(
    q, k, raw_gate, raw_beta, A_log, dt_bias, rstd, d_q_out, d_k_out, d_gate, d_beta, fastmath
):
    del rstd, d_q_out, d_k_out, d_gate, d_beta, fastmath
    inputs = (q, k, raw_gate, raw_beta, A_log, dt_bias)
    return tuple(torch.empty_like(t, memory_format=torch.contiguous_format) for t in inputs)


class _PrepareGDNInputs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, raw_gate, raw_beta, A_log, dt_bias, eps, fastmath):
        q_out, k_out, gate, beta, rstd = torch.ops.attn_gym.gdn_prepare_fwd(
            q, k, raw_gate, raw_beta, A_log, dt_bias, eps, fastmath
        )
        ctx.save_for_backward(q, k, raw_gate, raw_beta, A_log, dt_bias, rstd)
        ctx.fastmath = fastmath
        return q_out, k_out, gate, beta

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_q_out, d_k_out, d_gate, d_beta):
        # Autograd passes zeros, not None, for outputs the loss does not use.
        grads = torch.ops.attn_gym.gdn_prepare_bwd(
            *ctx.saved_tensors, d_q_out, d_k_out, d_gate, d_beta, ctx.fastmath
        )
        return *grads, None, None


def prepare_gdn_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    eps: float = 1e-6,
    fastmath: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(q, k, gate, beta)`` ready for ``chunk_gdn`` from raw projections, in one launch.

    Equivalent to ``l2norm(q)``, ``l2norm(k)``, ``gate_transform(raw_gate, A_log, dt_bias,
    kind="softplus", fastmath=fastmath)`` and ``sigmoid(raw_beta)``, with one forward and one
    backward kernel instead of four of each. As with ``gate_transform``, every token reaches the
    ``A_log``/``dt_bias`` gradients: callers with an inactive packed suffix mask the outputs.

    Args:
        q, k: Packed ``[1, T, HK, D]`` projections with a power-of-two ``D`` and last two strides
            ``(D, 1)``. Row-strided views of a fused ``[q|k|v]`` projection are read in place.
        raw_gate, raw_beta: Packed ``[1, T, H]`` decay and write-gate projections with
            contiguous heads.
        A_log, dt_bias: FP32 per-head parameters shaped ``[H]``.
        eps: L2-norm epsilon.
        fastmath: Approximate softplus/sigmoid primitives, as in ``gate_transform``.

    Example:
        q, k, gate, beta = prepare_gdn_inputs(q, k, a, b, A_log, dt_bias)
        out, _ = chunk_gdn(q, k, v, gate, beta, cu_seqlens=cu_seqlens)
    """
    inputs = (q, k, raw_gate, raw_beta, A_log, dt_bias)
    if not all(t.is_cuda and t.device == q.device for t in inputs):
        raise ValueError("all inputs must be CUDA tensors on one device")
    if q.ndim != 4 or q.shape[0] != 1 or k.shape != q.shape:
        raise ValueError("q and k must share a packed [1, T, HK, D] shape")
    _, tokens, _, head_dim = q.shape
    packed_heads = q.stride()[2:] == k.stride()[2:] == (head_dim, 1)
    if not packed_heads or triton.next_power_of_2(head_dim) != head_dim:
        raise ValueError("q and k need a power-of-two D and last two strides (D, 1)")
    heads = raw_gate.shape[-1]
    for name, gate in (("raw_gate", raw_gate), ("raw_beta", raw_beta)):
        if gate.shape != (1, tokens, heads) or gate.stride(2) != 1:
            raise ValueError(f"{name} must be [1, T, H] with contiguous heads, got {gate.shape}")
    for name, param in (("A_log", A_log), ("dt_bias", dt_bias)):
        if param.shape != (heads,) or param.dtype != torch.float32 or not param.is_contiguous():
            raise ValueError(f"{name} must be contiguous float32 with shape ({heads},)")
    return _PrepareGDNInputs.apply(q, k, raw_gate, raw_beta, A_log, dt_bias, eps, fastmath)


__all__ = ["prepare_gdn_inputs"]
