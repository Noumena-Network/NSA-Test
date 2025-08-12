#!/usr/bin/env python3
"""
Compare selection kernel output to reference implementation under deterministic indices.
"""

import math
import torch
import pytest

from nsa.kernels import (
    NSAConfig,
    SelectionAttention,
    derive_selection_from_compression_per_timestep,
    select_top_k_blocks_per_query_with_gqa,
)
from nsa.reference import selection_attention_reference


@pytest.mark.cuda
def test_selection_kernel_matches_reference():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    B, H, G, T = 2, 8, 2, 64
    dk, dv = 32, 32
    cfg = NSAConfig(
        head_dim_qk=dk,
        head_dim_v=dv,
        n_heads=H,
        n_kv_groups=G,
        l_prime=16,
        n=8,
        include_fixed_in_selection=False,
    )

    torch.manual_seed(0)
    q = torch.randn(B, H, T, dk, device=device, dtype=dtype)
    k = torch.randn(B, G, dk, T, device=device, dtype=dtype)
    v = torch.randn(B, G, T, dv, device=device, dtype=dtype)

    # Build deterministic selection indices by crafting monotonically increasing selection scores
    n_blocks = (T + cfg.l_prime - 1) // cfg.l_prime
    scores = torch.arange(n_blocks, device=device, dtype=dtype).view(1, 1, 1, n_blocks).expand(B, H, T, n_blocks)
    indices = select_top_k_blocks_per_query_with_gqa(scores, cfg)  # [B,G,T,n]

    # Kernel output via autograd wrapper for consistency
    o = SelectionAttention.apply(
        q, k, v, indices.to(torch.int32), 1.0 / math.sqrt(dk), cfg
    )

    # Reference output
    o_ref = selection_attention_reference(q, k, v, indices, cfg.l_prime, 1.0 / math.sqrt(dk))

    # Compare
    max_abs = (o - o_ref).abs().max().item()
    rel = (o - o_ref).abs().max() / (o_ref.abs().max() + 1e-8)
    assert max_abs < 1e-4 and rel < 1e-4, f"selection kernel vs reference mismatch: abs={max_abs}, rel={rel.item()}"
