"""prepare_gdn_inputs matches the unfused l2norm + gate_transform + sigmoid composition."""

import pytest
import torch

pytest.importorskip("triton")

from attn_gym.linear import gate_transform, l2norm
from attn_gym.linear.gdn import prepare_gdn_inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _inputs(tokens, key_heads, heads, *, fused_qkv):
    torch.manual_seed(0)
    device, dtype, head_dim = "cuda", torch.bfloat16, 128
    if fused_qkv:
        # Row-strided q/k views of one [q|k] projection, as a fused model produces them.
        qk = torch.randn(1, tokens, 2 * key_heads, head_dim, device=device, dtype=dtype)
        q, k = qk.split(key_heads, dim=2)
    else:
        q, k = torch.randn(2, 1, tokens, key_heads, head_dim, device=device, dtype=dtype)
    raw_gate = torch.randn(1, tokens, heads, device=device, dtype=dtype)
    raw_beta = torch.randn(1, tokens, heads, device=device, dtype=dtype)
    A_log = torch.randn(heads, device=device)
    dt_bias = torch.randn(heads, device=device)
    return [t.detach().requires_grad_() for t in (q, k, raw_gate, raw_beta, A_log, dt_bias)]


# T=1000 leaves a partial token block; H=48 leaves masked lanes in the power-of-two head tile.
@pytest.mark.parametrize(
    ("fused_qkv", "heads", "fastmath"), [(False, 32, True), (True, 48, False)]
)
def test_matches_unfused(fused_qkv, heads, fastmath):
    leaves = _inputs(1000, 16, heads, fused_qkv=fused_qkv)
    q, k, raw_gate, raw_beta, A_log, dt_bias = leaves
    fused = prepare_gdn_inputs(*leaves, fastmath=fastmath)
    unfused = (
        l2norm(q),
        l2norm(k),
        gate_transform(raw_gate, A_log, dt_bias, kind="softplus", fastmath=fastmath),
        # FP32 like the kernel: a BF16 sigmoid backward reuses the rounded output.
        torch.sigmoid(raw_beta.float()).to(raw_beta.dtype),
    )
    cotangents = [torch.randn_like(t) for t in unfused]
    actual = [*fused, *torch.autograd.grad(fused, leaves, cotangents)]
    expected = [*unfused, *torch.autograd.grad(unfused, leaves, cotangents)]
    names = ["q", "k", "gate", "beta"]
    names += ["dq", "dk", "d_raw_gate", "d_raw_beta", "dA_log", "d_dt_bias"]
    for name, got, want in zip(names, actual, expected, strict=True):
        # Same FP32 math; only reduction order and beta's fastmath sigmoid differ.
        torch.testing.assert_close(got, want, rtol=1.6e-2, atol=1e-4, msg=name)


def test_ops_pass_opcheck():
    inputs = [t.detach() for t in _inputs(64, 2, 4, fused_qkv=True)]
    fwd_args = (*inputs, 1e-6, False)
    torch.library.opcheck(torch.ops.attn_gym.gdn_prepare_fwd.default, fwd_args)
    *outputs, rstd = torch.ops.attn_gym.gdn_prepare_fwd(*fwd_args)
    bwd_args = (*inputs, rstd, *(torch.randn_like(t) for t in outputs), False)
    torch.library.opcheck(torch.ops.attn_gym.gdn_prepare_bwd.default, bwd_args)
