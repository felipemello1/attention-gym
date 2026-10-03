"""Gather attention over explicit KV indices and an optional causal local window."""

from attn_gym.types import Impl

from .api import AuxRequest, GatherAttnAux, gather_attn

# Define the CuTe operators at import, like the indexer and linear operators, so callers can
# name attn_gym::_gather_attn_cute_fwd (e.g. in an activation-checkpointing save list) before
# the first call. impl/cute.py, which imports torch._dynamo and FA4, still loads on first use.
from .impl import cute_ops as _cute_ops

__all__ = ["AuxRequest", "GatherAttnAux", "Impl", "gather_attn"]
