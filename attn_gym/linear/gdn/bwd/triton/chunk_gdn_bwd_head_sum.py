# Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fold expanded-head FP32 Q/K gradient pieces into grouped low-precision gradients."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sum_expanded_head_gradients_kernel(
    first,
    second,
    out,
    num_rows,
    GROUPS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program folds BLOCK_ROWS (token, key head) rows; int64 offsets keep long packed
    # batches safe.
    rows = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    columns = tl.arange(0, BLOCK_K)
    mask = (rows < num_rows)[:, None] & (columns < K)[None, :]
    accumulator = tl.zeros([BLOCK_ROWS, BLOCK_K], dtype=tl.float32)
    for group in tl.static_range(GROUPS):
        offsets = (rows[:, None] * GROUPS + group) * K + columns[None, :]
        accumulator += tl.load(first + offsets, mask=mask, other=0.0) + tl.load(
            second + offsets, mask=mask, other=0.0
        )
    tl.store(
        out + rows[:, None] * K + columns[None, :],
        accumulator.to(out.dtype.element_ty),
        mask=mask,
    )


def sum_expanded_head_gradients(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    groups: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Add two expanded-head FP32 gradients, sum each key head's group, and cast once.

    Equivalent to ``(first + second).unflatten(2, (-1, groups)).sum(3).to(dtype)`` in one
    pass instead of an add, a reduction, and a cast over ``[B, T, H, K]`` FP32 buffers.

    Args:
        first: Contiguous FP32 ``[B, T, H, K]`` gradient over the expanded ``H`` heads.
        second: Contiguous FP32 gradient with the same shape as ``first``.
        groups: Value heads per key head; ``H`` must be divisible by it.
        dtype: Output dtype.

    Returns:
        ``[B, T, H // groups, K]`` gradient in ``dtype``.

    Example:

        first, second: [1, T, 32, 128] FP32, groups=2  ->  [1, T, 16, 128] BF16
    """
    batch, tokens, heads, head_dim = first.shape
    assert second.shape == first.shape and heads % groups == 0
    assert first.dtype == second.dtype == torch.float32
    assert first.is_contiguous() and second.is_contiguous()
    out = first.new_empty(batch, tokens, heads // groups, head_dim, dtype=dtype)
    num_rows = batch * tokens * (heads // groups)
    if num_rows == 0:
        return out
    # Four rows per program reach ~6 TB/s on GB300 for 2-3 heads per group; one row per
    # program leaves the load pipeline underfed (~4.4 TB/s at 2 heads per group).
    block_rows = 4
    _sum_expanded_head_gradients_kernel[(triton.cdiv(num_rows, block_rows),)](
        first,
        second,
        out,
        num_rows,
        GROUPS=groups,
        K=head_dim,
        BLOCK_ROWS=block_rows,
        BLOCK_K=triton.next_power_of_2(head_dim),
        num_warps=8,
    )
    return out


__all__ = ["sum_expanded_head_gradients"]
