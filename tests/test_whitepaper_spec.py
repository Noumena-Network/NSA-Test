#!/usr/bin/env python3
"""
Whitepaper specification conformance tests for NSA.
Focus on Eq. 5 (gating), Eq. 8→9→10→11 (selection), and compression interfaces.
"""

import math
import torch
import pytest

from nsa.kernels import (
    NSAConfig,
    NSAAttention,
    derive_selection_from_compression_per_timestep,
    select_top_k_blocks_per_query_with_gqa,
)


@pytest.mark.cuda
def test_gate_mode_defaults_and_softmax_sum():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = NSAConfig(d_model=256)  # defaults gate_mode='sigmoid'
    assert cfg.gate_mode == 'sigmoid'

    cfg_soft = NSAConfig(d_model=256, gate_mode='softmax')
    model = NSAAttention(cfg_soft).to(device)
    model.eval()

    B, T = 2, 32
    hidden = torch.randn(B, T, cfg_soft.d_model, device=device)
    with torch.no_grad():
        out, info = model(hidden, output_attentions=True)
    # Gates shape [B, T, H, 3] and sum to ~1 along last dim
    gates = info['gates']  # [B, T, H, 3]
    sums = gates.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-6)


def test_selection_topn_paper_faithful_and_fixed_toggle():
    # Build deterministic selection scores where aggregated scores are strictly increasing per block
    B, T = 1, 3
    G, H = 1, 2  # heads_per_group = 2
    n_blocks = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # selection_scores: [B, H, T, n_blocks]
    base = torch.arange(n_blocks, device=device, dtype=torch.float32)
    V = base.view(1, 1, 1, n_blocks)
    selection_scores = V.repeat(B, H, T, 1)  # all heads identical

    # Paper-faithful: no fixed blocks
    cfg = NSAConfig(l_prime=2, d=1, n=3, n_heads=H, n_kv_groups=G, include_fixed_in_selection=False)
    idx = select_top_k_blocks_per_query_with_gqa(selection_scores, cfg)
    # Expect top 3 blocks by value: [4,3,2]
    expected = torch.tensor([4, 3, 2], device=device)
    for t in range(T):
        assert torch.equal(idx[0, 0, t], expected)

    # With fixed blocks enabled, dedup preserving order to n
    cfg_fixed = NSAConfig(l_prime=2, d=1, n=3, n_heads=H, n_kv_groups=G, include_fixed_in_selection=True, n_fixed=3)
    idx_fixed = select_top_k_blocks_per_query_with_gqa(selection_scores, cfg_fixed)
    # t=0 current block=0, expect [0,4,3]
    assert torch.equal(idx_fixed[0, 0, 0], torch.tensor([0, 4, 3], device=device))
    # t=2 current block=1, expect includes 0 and 1, then the next highest (4)
    assert torch.equal(idx_fixed[0, 0, 2], torch.tensor([0, 1, 4], device=device))


def test_selection_padding_when_n_exceeds_blocks():
    B, T = 1, 2
    G, H = 1, 2
    n_blocks = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scores = torch.zeros(B, H, T, n_blocks, device=device)
    cfg = NSAConfig(l_prime=2, d=1, n=4, n_heads=H, n_kv_groups=G)
    idx = select_top_k_blocks_per_query_with_gqa(scores, cfg)
    # Expect length n with -1 padding
    assert idx.shape[-1] == 4
    assert (idx[0, 0, 0, 2:] == -1).all()


def test_compress_tokens_interfaces_and_block_ends():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = NSAConfig(d_model=256, l=8, d=4, n_heads=4, n_kv_groups=2)
    model = NSAAttention(cfg).to(device)
    B, T = 1, 24
    dk, dv = cfg.head_dim_qk, cfg.head_dim_v
    # Fake K/V per branch input shape [B, G, T, d{qk/v}]
    keys = torch.randn(B, cfg.n_kv_groups, T, dk, device=device)
    vals = torch.randn(B, cfg.n_kv_groups, T, dv, device=device)

    k_cmp, v_cmp, block_ends, n_actual = model.compress_tokens(keys, vals)
    # Shapes match spec
    assert k_cmp.shape[:3] == (B, cfg.n_kv_groups, dk)
    assert v_cmp.shape[:2] == (B, cfg.n_kv_groups)
    # Block ends follow i*d + (l-1) for real blocks
    expected_n = (T - cfg.l) // cfg.d + 1
    assert n_actual == expected_n
    for i in range(n_actual):
        assert int(block_ends[i].item()) == i * cfg.d + cfg.l - 1
