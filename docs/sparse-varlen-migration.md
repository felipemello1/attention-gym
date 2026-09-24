# Packed documents in gather attention and the lightning indexer

`gather_attn` replaces its `doc_ids` argument with paired `cu_seqlens` and
`cu_seqlens_k` offsets. `lightning_indexer` accepts the same pair. Packed calls
use the existing tensor layouts with batch size one; neither operation accepts
packed 3-D queries. Gather's `sliding_window_size` is now keyword-only.

| Tensor | Gather attention | Lightning indexer |
| --- | --- | --- |
| Query | `[1, H, T, D]` | `[1, T, H, D]` |
| Local KV | `[1, Hkv, T, D]` | — |
| Sparse/candidate KV | `[1, Hkv, S, D]` | `[1, S, D]` |
| Selected indices | Input `[1, T, topk]` | Output `[1, T, topk]` |

Without offsets, ordinary batched behavior is unchanged. With offsets:

- `cu_seqlens` partitions queries and gather's local KV into documents.
- `cu_seqlens_k` partitions the sparse/candidate KV pool into the same documents.
- Both are contiguous `int32` tensors on the query device, shaped `[N + 1]`.
  They start at zero, never decrease, and may repeat for empty documents. Their
  final offsets may be below the allocated T/S capacities. These value
  properties are caller invariants; validation does not synchronize to read them.
- Selected indices are **zero-based within the query's document's sparse pool**.
  `-1` means no selection. Gather ignores out-of-document local selections;
  duplicates retain their multiplicity in attention. The sliding window is
  generated internally and is not listed in these indices.
- The indexer excludes other documents **before** top-k. Missing selections are
  padded with `-1`, even when `topk > S` or the entire candidate pool is empty.
- Inactive query-capacity rows return `-1` from the indexer. Gather output and
  token gradients beyond the query endpoint are undefined; fixed-capacity
  callers must mask those rows, as with KDA.

The fused gather paths do not allocate per-token document labels. Triton reads
`cu_seqlens` directly to bound local attention in forward and backward. The
CuTe/FA4 adapter uses those same offsets when building FA4's existing gather-index
tensor; FA4 itself is unchanged. Do not modify query offsets between forward and
backward, since they define the attention operation whose gradients are computed.

## Compression boundaries

The indexer processes whole documents starting at position zero. At compression
ratio `r`, document `d` must contain `length[d] // r` candidates. Each document's
incomplete final group produces no candidate; its tokens remain queries and
local KV. Do not carry a remainder into the next document.

For lengths `[3, 5, 7]` at ratio 4:

```text
cu_seqlens   = [0, 3, 8, 15]
cu_seqlens_k = [0, 0, 1, 2]
```

There are two complete compressed blocks, not `15 // 4 == 3`. Derive offsets
once from **per-document** lengths and reuse them across both operations:

```python
compressed_lengths = cu_seqlens.diff() // compress_ratio
cu_seqlens_k = torch.cat(
    [cu_seqlens.new_zeros(1), compressed_lengths.cumsum(0, dtype=torch.int32)]
)
```

This describes the physical candidate pool; the producer must actually pack
that pool in the corresponding document order. Computing new offsets cannot
repair KV that was already compressed across document boundaries.

## TorchTitan DSv4 migration

1. Replace the `attn_gym.sparse.selected_attention` import/call with
   `attn_gym.sparse.gather_attn.gather_attn`. Keep the existing THD-to-4-D adapters:
   gather queries use `q.transpose(0, 1).unsqueeze(0)`; indexer queries use
   `q.unsqueeze(0)`.
2. Pass query offsets from `VarlenMetadata.cu_seq_q` into both operations.
   Build **compressed** offsets for the candidate pool; do not blindly reuse
   uncompressed `VarlenMetadata.cu_seq_k` as `cu_seqlens_k`.
3. Update both the attention and indexer compressors in
   `torchtitan/models/deepseek_v4/compressor.py`: restart grouping, learned
   intra-group positions, previous-group overlap, and compressed RoPE positions
   at each document boundary. Emit only complete groups within each document.
4. Pass the same offsets to `Indexer.select` and gather. Indexer output now
   feeds gather directly, without adding packed document bases. Replace global
   HCA causal limits in `attention.py` with document-local limits as well.
5. Update auxiliary losses or other consumers that gather directly from packed
   sparse tensors: translate each valid local index with
   `cu_seqlens_k[document] + index`, while preserving `-1` and masking before
   accessing storage. No translation is needed before calling `gather_attn`.

These stateless operations do not add cached decoding or chunked-prefill
support. In an inference compressor, a chunk boundary is **not** a document
boundary: retain partial-group state when the same request continues. Query
storage offsets alone do not describe a query chunk's position in its request's
causal timeline.
