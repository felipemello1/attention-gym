"""Gather attention over explicit KV indices and an optional causal local window."""

from attn_gym.types import Impl

from .api import AuxRequest, GatherAttnAux, gather_attn

# Register the CuTe operators at import, like the indexer and linear operators, so callers can
# name attn_gym::_gather_attn_cute_fwd (e.g. in an activation-checkpointing save list) before
# the first call. The module imports FA4 only when a kernel runs.
from .impl import cute as _cute

__all__ = ["AuxRequest", "GatherAttnAux", "Impl", "gather_attn"]
