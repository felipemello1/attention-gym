"""Fused expanded-head gradient fold used by the Blackwell GDN backward."""

import pytest
import torch

pytest.importorskip("triton")

from attn_gym.linear.gdn.bwd.triton.chunk_gdn_bwd_head_sum import sum_expanded_head_gradients

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("tokens", [1, 77, 4096])
@pytest.mark.parametrize(("heads", "groups"), [(16, 1), (32, 2), (48, 3)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matches_add_group_sum_cast(tokens: int, heads: int, groups: int, dtype: torch.dtype):
    first = torch.randn(1, tokens, heads, 128, device="cuda")
    second = torch.randn_like(first)
    expected = (first + second).unflatten(2, (-1, groups)).sum(3).to(dtype)

    actual = sum_expanded_head_gradients(first, second, groups=groups, dtype=dtype)

    assert actual.shape == (1, tokens, heads // groups, 128)
    # FP32 summation order may differ from torch's reduction by one rounding before the cast.
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-2)
    exact = (actual == expected).float().mean().item()
    assert exact > 0.99


def test_empty_batch():
    first = torch.empty(1, 0, 32, 128, device="cuda")
    out = sum_expanded_head_gradients(first, first.clone(), groups=2, dtype=torch.bfloat16)
    assert out.shape == (1, 0, 16, 128)
