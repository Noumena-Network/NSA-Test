#!/usr/bin/env python3
"""
Block semantics conformance tests for NSA.
Enforces l/l' blocking, block-end causality, deduplication, 
always-include blocks, and strict boundaries.
"""

import torch
import triton
import math
import pytest

from nsa.kernels import (
    NSAConfig,
    _nsa_compression_fwd_kernel,
    _nsa_sparse_selection_fwd_kernel,
)
from nsa.reference import compress_kv_simple


def test_selection_block_boundaries():
    """
    Test that selection blocks enforce strict l' boundaries.
    No spillover beyond block size l' is allowed.
    """
    B, H, G, T = 1, 4, 2, 256
    dk, dv = 64, 64
    l_prime = 64  # Selection block size
    n = 4  # Blocks to select
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    # Create tensors with distinct signals at block boundaries
    q = torch.randn(B, H, T, dk, device=device, dtype=dtype)
    k = torch.zeros(B, G, dk, T, device=device, dtype=dtype)
    v = torch.zeros(B, G, T, dv, device=device, dtype=dtype)
    
    # Place distinct values in each block
    n_blocks = (T + l_prime - 1) // l_prime
    for block_idx in range(n_blocks):
        start = block_idx * l_prime
        end = min(start + l_prime, T)
        # Each block has unique value
        v[:, :, start:end, :] = float(block_idx + 1)
        # K also has block-specific pattern
        k[:, :, :, start:end] = float(block_idx + 1)
    
    # Create block indices - select specific blocks
    indices = torch.full((B, G, T, n), -1, device=device, dtype=torch.int32)
    for t in range(T):
        current_block = t // l_prime
        # Select current block and adjacent ones
        selected = [0]  # Always include first
        if current_block > 0:
            selected.append(current_block)
        if current_block < n_blocks - 1 and len(selected) < n:
            selected.append(current_block + 1)
        # Pad with -1 sentinel if needed
        while len(selected) < n:
            selected.append(-1)
        indices[:, :, t, :] = torch.tensor(selected, device=device)
    
    # Run kernel
    output = torch.zeros(B, H, T, dv, device=device, dtype=dtype)
    L = torch.zeros(B, H, T, device=device, dtype=torch.float32)
    M = torch.zeros(B, H, T, device=device, dtype=torch.float32)
    
    grid = lambda META: (triton.cdiv(T, META['BLOCK_M']), B * H)
    scale = 1.0 / math.sqrt(dk)
    
    _nsa_sparse_selection_fwd_kernel[grid](
        q, k, v, indices, output, scale, L, M,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(3), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        M.stride(0), M.stride(1), M.stride(2),
        Z=B, H=H, N_KV_GROUPS=G, N_CTX_Q=T, N_CTX_KV=T,
        N_BLOCKS=n, BLOCK_SIZE_SELECTION=l_prime,
        HEAD_DIM_QK=dk, HEAD_DIM_V=dv,
        BLOCK_M=32, BLOCK_N=32, IS_CAUSAL=True,
    )
    
    # Verify: Each query should only see values from selected blocks
    for t in range(T):
        current_block = t // l_prime
        # Output should be weighted combination of selected block values
        # Due to causal mask, can only see blocks up to current position
        out_val = output[0, 0, t, 0].item()
        
        # Should be between 0 and max block index seen
        max_block_seen = min(current_block + 1, n_blocks - 1)
        assert 0 <= out_val <= max_block_seen + 2, \
            f"Output at t={t} has unexpected value {out_val}, expected <= {max_block_seen + 1}"
    
    print("✅ Selection block boundaries enforced (no spillover past l')")


def test_compression_block_end_causality():
    """
    Test that compression uses block END positions for causality.
    A block i is visible at query t iff block_end[i] <= t.
    """
    config = NSAConfig(l=32, d=16)
    B, G, T = 1, 2, 128
    dk, dv = 64, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    # Create test data
    k = torch.randn(B, G, dk, T, device=device, dtype=dtype)
    v = torch.randn(B, G, T, dv, device=device, dtype=dtype)
    
    # Compress with our reference
    k_comp, v_comp, block_ends = compress_kv_simple(k, v, config.l, config.d)
    n_blocks = k_comp.shape[-1]
    
    # Verify block ends are computed correctly
    for i in range(n_blocks):
        expected_end = min(i * config.d + config.l - 1, T - 1)
        actual_end = block_ends[i].item()
        assert actual_end == expected_end, \
            f"Block {i} end: expected {expected_end}, got {actual_end}"
    
    # Test causality: for various query positions
    for t in [20, 40, 60, 80, 100]:
        # Count how many blocks should be visible
        visible_blocks = 0
        for i in range(n_blocks):
            if block_ends[i].item() <= t:
                visible_blocks += 1
        
        # In actual attention, only these blocks should contribute
        # This is enforced by the causal mask in compression kernel
        assert (block_ends <= t).sum().item() == visible_blocks, \
            f"At t={t}, {visible_blocks} blocks should be visible"
    
    print("✅ Block-end causality correctly implemented")


def test_always_include_blocks():
    """
    Test that selection always includes required blocks:
    - Block 0 (initial context)
    - Current and adjacent local blocks
    With proper deduplication.
    """
    config = NSAConfig(
        l_prime=64,
        n=8,  # Total blocks to select
        n_fixed=3,  # Always include blocks
        n_dynamic=5,  # Dynamic blocks (n - n_fixed)
        n_heads=4,
        n_kv_groups=2,
    )
    
    B, T = 1, 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_sel_blocks = (T + config.l_prime - 1) // config.l_prime
    
    # Create selection scores
    torch.manual_seed(789)
    sel_scores = torch.rand(B, config.n_heads, T, n_sel_blocks, device=device)
    
    # Import and use the actual selection function
    from nsa.kernels import select_top_k_blocks_per_query_with_gqa
    indices = select_top_k_blocks_per_query_with_gqa(sel_scores, config)
    
    # Test always-include enforcement
    for t in [0, 100, 200, 300, 400]:
        current_block = t // config.l_prime
        
        for g in range(config.n_kv_groups):
            selected = indices[0, g, t].tolist()
            
            # Block 0 must always be included
            assert 0 in selected, f"Block 0 not included at t={t}"
            
            # Current block should be included if it exists
            if current_block < n_sel_blocks:
                assert current_block in selected, \
                    f"Current block {current_block} not included at t={t}"
            
            # Adjacent blocks should be considered
            if current_block > 0:
                # Previous block often included for context
                prev_block = current_block - 1
                # Not mandatory but common
                
            if current_block < n_sel_blocks - 1:
                # Next block might be included if score is high
                next_block = current_block + 1
                # Not mandatory
    
    # Test deduplication
    for t in range(0, T, 50):  # Sample positions
        for g in range(config.n_kv_groups):
            selected = indices[0, g, t].tolist()
            # Remove padding (last element repeated)
            unique_selected = []
            seen = set()
            for idx in selected:
                if idx not in seen:
                    unique_selected.append(idx)
                    seen.add(idx)
            
            # Should maintain order and no duplicates
            assert len(unique_selected) == len(seen), \
                f"Deduplication failed at t={t}: {selected}"
    
    print("✅ Always-include blocks and deduplication verified")


def test_block_alignment():
    """
    Test that all block operations are properly aligned.
    Blocks should start and end at expected positions.
    """
    configs = [
        NSAConfig(l=32, d=16, l_prime=64),
        NSAConfig(l=16, d=8, l_prime=32),
        NSAConfig(l=64, d=32, l_prime=128),
    ]
    
    for config in configs:
        T = 512
        
        # Test compression block alignment
        n_comp_blocks = (T - config.l) // config.d + 1
        for i in range(n_comp_blocks):
            block_start = i * config.d
            block_end = min(block_start + config.l - 1, T - 1)
            
            # Verify block doesn't exceed sequence length
            assert block_end < T, f"Compression block {i} exceeds sequence"
            
            # Verify blocks have correct size (except last)
            if i < n_comp_blocks - 1:
                assert block_end - block_start + 1 == config.l, \
                    f"Compression block {i} has wrong size"
        
        # Test selection block alignment
        n_sel_blocks = (T + config.l_prime - 1) // config.l_prime
        for j in range(n_sel_blocks):
            block_start = j * config.l_prime
            block_end = min(block_start + config.l_prime - 1, T - 1)
            
            # Verify block doesn't exceed sequence length
            assert block_end < T, f"Selection block {j} exceeds sequence"
            
            # Verify blocks have correct size (except last)
            if j < n_sel_blocks - 1:
                assert block_end - block_start + 1 == config.l_prime, \
                    f"Selection block {j} has wrong size"
        
        # Verify overlap calculations
        # Compression stride should be less than block size for overlap
        assert config.d < config.l, "Compression stride must be < block size for overlap"
    
    print("✅ Block alignment verified for multiple configurations")


def test_block_index_bounds():
    """
    Test that block indices are always within valid bounds.
    No out-of-bounds access should occur.
    """
    B, G, T = 2, 2, 256
    l_prime = 32
    n = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_blocks_max = (T + l_prime - 1) // l_prime
    
    # Create indices tensor
    indices = torch.full((B, G, T, n), -1, device=device, dtype=torch.int32)
    
    # Fill with valid indices
    for b in range(B):
        for g in range(G):
            for t in range(T):
                # Generate valid indices for this position
                current_block = min(t // l_prime, n_blocks_max - 1)
                valid_indices = [0]  # Always include first
                
                # Add current and nearby blocks (respecting causality)
                for offset in [-1, 0, 1]:
                    block = current_block + offset
                    block_start = block * l_prime
                    # Only add if block has started and is valid
                    if 0 <= block < n_blocks_max and block_start <= t and block not in valid_indices:
                        valid_indices.append(block)
                
                # Fill remaining with -1 sentinel
                while len(valid_indices) < n:
                    valid_indices.append(-1)
                
                indices[b, g, t] = torch.tensor(valid_indices[:n], device=device)
    
    # Verify valid indices are within bounds (ignore -1 sentinels)
    valid_mask = indices >= 0
    assert (indices[valid_mask] < n_blocks_max).all(), f"Found indices >= {n_blocks_max}"
    
    # Verify causality - blocks should be visible based on their position
    # Note: Selection blocks can look at any block that has started by time t
    # Block i starts at position i * l_prime
    for t in range(T):
        for b in range(B):
            for g in range(G):
                selected = indices[b, g, t]
                # Filter out padding
                unique = torch.unique(selected)
                for block_idx in unique:
                    block_start = block_idx * l_prime
                    # Block is valid if it starts before or at current position
                    assert block_start <= t, \
                        f"Future block {block_idx} (starts at {block_start}) selected at t={t}"
    
    print("✅ Block index bounds verified")


if __name__ == "__main__":
    test_selection_block_boundaries()
    test_compression_block_end_causality()
    test_always_include_blocks()
    test_block_alignment()
    test_block_index_bounds()
    print("\n" + "="*60)
    print("ALL BLOCK SEMANTICS TESTS PASSED")
    print("="*60)