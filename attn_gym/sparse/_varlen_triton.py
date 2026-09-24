"""Prepare packed candidate intervals inside the indexer operator."""

import torch
import triton
import triton.language as tl
from torch import Tensor

from attn_gym._backends.triton.utils import requires_int64_offsets


@triton.jit
def _document_ids(CuQ, query, num_documents, WIDE: tl.constexpr):
    low = tl.full(query.shape, 0, tl.int32)
    high = tl.full(query.shape, num_documents, tl.int32)
    remaining = num_documents
    # A runtime logarithmic loop works for both scalar queries and vectorized bounds.
    # Right-sided search skips empty documents and returns N for inactive capacity.
    while remaining > 0:
        middle = low + (high - low) // 2
        offset = middle.to(tl.int64) if WIDE else middle
        boundary = tl.load(CuQ + offset + 1, middle < num_documents, 0)
        right = query >= boundary
        active = low < high
        low = tl.where(active & right, middle + 1, low)
        high = tl.where(active & ~right, middle, high)
        remaining = remaining // 2
    return low


@triton.jit
def _candidate_bounds_kernel(
    CuQ,
    CuK,
    Bounds,
    num_documents,
    tokens,
    CAUSAL: tl.constexpr,
    RATIO: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    if WIDE:
        block = block.to(tl.int64)
    query = block * BLOCK + tl.arange(0, BLOCK)
    document = _document_ids(CuQ, query, num_documents, WIDE)
    offset = document.to(tl.int64) if WIDE else document
    start = tl.load(CuK + offset)
    end = tl.load(CuK + tl.minimum(offset + 1, num_documents))
    if CAUSAL:
        local_position = query - tl.load(CuQ + offset)
        end = tl.minimum(end, start + (local_position + 1) // RATIO)
    tl.store(Bounds + query * 2, start, query < tokens)
    tl.store(Bounds + query * 2 + 1, end, query < tokens)


def prepare_candidate_bounds(
    cu_seqlens: Tensor,
    cu_seqlens_k: Tensor,
    tokens: int,
    causal: bool,
    compress_ratio: int,
) -> Tensor:
    bounds = cu_seqlens.new_empty((tokens, 2))
    if tokens == 0:
        return bounds
    with torch.cuda.device(cu_seqlens.device):
        _candidate_bounds_kernel[(triton.cdiv(tokens, 256),)](
            cu_seqlens,
            cu_seqlens_k,
            bounds,
            cu_seqlens.numel() - 1,
            tokens,
            causal,
            compress_ratio,
            requires_int64_offsets(cu_seqlens, cu_seqlens_k, bounds),
            256,
            num_warps=4,
        )
    return bounds
