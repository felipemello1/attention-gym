"""Packed-document metadata shared by sparse attention and candidate selection."""

import torch
from torch import Tensor


def validate_packed_sequences(
    cu_seqlens: Tensor | None,
    cu_seqlens_k: Tensor | None,
    *,
    batch: int,
    device: torch.device,
) -> None:
    """Validate metadata only; offset values are a caller contract, as in KDA."""
    if (cu_seqlens is None) != (cu_seqlens_k is None):
        raise ValueError("cu_seqlens and cu_seqlens_k must be supplied together")
    if cu_seqlens is None:
        return
    if batch != 1:
        raise ValueError("packed cu_seqlens require q to have batch size one")
    for name, offsets in (("cu_seqlens", cu_seqlens), ("cu_seqlens_k", cu_seqlens_k)):
        if not isinstance(offsets, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if offsets.ndim != 1 or offsets.shape[0] < 2:
            raise ValueError(f"{name} must have shape [num_sequences + 1]")
        if offsets.dtype != torch.int32 or not offsets.is_contiguous() or offsets.device != device:
            raise ValueError(f"{name} must be contiguous int32 on q.device")
    if cu_seqlens.shape != cu_seqlens_k.shape:
        raise ValueError("cu_seqlens and cu_seqlens_k must describe the same number of sequences")


def packed_sequence_metadata(
    cu_seqlens: Tensor, cu_seqlens_k: Tensor, tokens: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Return local query positions and packed candidate start/end per token.

    Right-sided search skips repeated offsets for empty documents. Capacity-tail tokens map
    to the terminal offset and an empty candidate interval, without reading beyond either
    prefix tensor. All returned tensors have shape [tokens] and dtype int32.
    """
    positions = torch.arange(tokens, device=cu_seqlens.device, dtype=torch.int32)
    documents = torch.searchsorted(cu_seqlens[1:], positions, right=True, out_int32=True)
    local_positions = positions - cu_seqlens.index_select(0, documents)
    starts = cu_seqlens_k.index_select(0, documents)
    ends = cu_seqlens_k.index_select(0, (documents + 1).clamp(max=cu_seqlens_k.shape[0] - 1))
    return local_positions, starts, ends
