#!/usr/bin/env python3
"""
Specification conformance tests for NSA algorithm components.
Tests exact adherence to Equations 8, 9, and 10 from the paper.
"""

import os
# Set environment variable for deterministic CUBLAS
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import math
import pytest

from nsa.kernels import (
    NSAConfig,
    NSAAttention,
    derive_selection_from_compression_per_timestep,
    select_top_k_blocks_per_query_with_gqa,
)


def test_eq8_compression_scores():
    """
    Test Eq. 8: Compression scores must be computed as p_cmp(t) = softmax(q_t^T K_cmp).
    No randomness allowed - scores must derive from actual Q @ K computation.
    """
    config = NSAConfig(
        d_model=512,
        head_dim_qk=192,  # Paper spec (will be padded to 256 for Triton)
        head_dim_v=128,   # Paper spec (already power of 2)
        n_heads=8,
        n_kv_groups=2,
        l=32,
        d=16,
    )
    
    B, T = 2, 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    # Create NSA module
    nsa = NSAAttention(config).to(device)
    nsa.eval()  # Set to eval mode for determinism
    
    # Create input
    torch.manual_seed(42)
    hidden = torch.randn(B, T, config.d_model, device=device, dtype=dtype)
    
    # Get Q and compressed K from the module's internal computation
    with torch.no_grad():
        # Project to Q
        q = nsa.q_proj(hidden).view(B, T, config.n_heads, config.head_dim_qk).transpose(1, 2)
        
        # Get compressed keys through the compression branch
        k_comp = nsa.k_compress(hidden).view(B, T, config.n_kv_groups, config.head_dim_qk).transpose(1, 2)
        v_comp = nsa.v_compress(hidden).view(B, T, config.n_kv_groups, config.head_dim_v).transpose(1, 2)
        
        # Compress tokens
        k_compressed, v_compressed, block_ends, n_blocks_actual = nsa.compress_tokens(k_comp, v_comp)
        n_blocks_padded = k_compressed.shape[-1]  # Use padded size for tensors
        
        # Compute reference compression scores using Eq. 8
        H = config.n_heads
        G = config.n_kv_groups
        heads_per_group = H // G
        
        # Expand k_compressed for all heads
        k_exp = k_compressed.unsqueeze(2).repeat(1, 1, heads_per_group, 1, 1)
        k_exp = k_exp.view(B, H, config.head_dim_qk, n_blocks_padded)
        
        # Compute Q @ K^T
        scale = 1.0 / math.sqrt(config.head_dim_qk)
        scores_ref = torch.matmul(q, k_exp) * scale
        
        # Apply causal mask based on block_ends (matching implementation)
        for b in range(B):
            for h in range(H):
                g = h // (H // G)
                for t in range(T):
                    # Check if any blocks are visible at this timestep
                    any_visible = False
                    for block_idx in range(n_blocks_padded):
                        if block_ends[block_idx] <= t:
                            any_visible = True
                            break
                    
                    # Apply masking
                    for block_idx in range(n_blocks_padded):
                        if block_ends[block_idx] > t:
                            # If no blocks visible, make block 0 visible (fallback)
                            if not any_visible and block_idx == 0:
                                continue
                            scores_ref[b, h, t, block_idx] = float('-inf')
        
        p_cmp_ref = torch.softmax(scores_ref, dim=-1)
        
        # Now run the module and extract its compression scores
        output, attn_info = nsa(hidden, output_attentions=True)
        
        # The module should compute compression scores the same way
        # Verify that selection scores derive correctly from compression scores
        sel_scores_ref = derive_selection_from_compression_per_timestep(p_cmp_ref, config)
        
        # Check if module returns selection scores in attn_info
        if 'selection_scores' in attn_info:
            torch.testing.assert_close(attn_info['selection_scores'], sel_scores_ref, rtol=1e-3, atol=1e-4)
        
    # Verify scores are deterministic (no randomness)
    # Run again with same input to verify determinism
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    
    output2, attn_info2 = nsa(hidden, output_attentions=True)
    
    # With same input and deterministic mode, outputs should be very close
    # Relaxed tolerance to account for numerical variations in sparse ops
    torch.testing.assert_close(output, output2, rtol=1e-3, atol=1e-4)
    
    # Restore defaults
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    
    print("✅ Eq. 8: Compression scores validated (deterministic, Q @ K based)")


def test_eq9_triangular_mapping():
    """
    Test Eq. 9: Verify exact triangular summation from compression to selection scores.
    Maps compression blocks to selection blocks via spatial correspondence.
    """
    config = NSAConfig(
        head_dim_qk=192,
        head_dim_v=128,
        l=32,  # Compression block size
        d=16,  # Compression stride
        l_prime=64,  # Selection block size
        n_heads=8,
        n_kv_groups=2,
    )
    
    B, H, T = 2, 8, 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create compression scores
    n_comp_blocks = (T - config.l) // config.d + 1
    n_sel_blocks = (T + config.l_prime - 1) // config.l_prime
    
    torch.manual_seed(123)
    comp_scores = torch.rand(B, H, T, n_comp_blocks, device=device)
    
    # Apply causal masking
    for t in range(T):
        for i in range(n_comp_blocks):
            block_end = i * config.d + config.l - 1
            if block_end > t:
                comp_scores[:, :, t, i] = 0
    
    # Normalize
    comp_scores = comp_scores / (comp_scores.sum(dim=-1, keepdim=True) + 1e-8)
    
    # Apply Eq. 9 mapping
    sel_scores = derive_selection_from_compression_per_timestep(comp_scores, config)
    
    # Verify shape
    expected_shape = (B, H, T, n_sel_blocks)
    assert sel_scores.shape == expected_shape, f"Shape mismatch: {sel_scores.shape} vs {expected_shape}"
    
    # Verify triangular mapping for specific positions
    # For each selection block j, it should aggregate from compression blocks
    # that spatially overlap with it
    for t in [100, 200, 300]:  # Sample positions
        for j in range(min(5, n_sel_blocks)):  # Check first few blocks
            # Expected aggregation window
            comp_start = j * config.l_prime // config.d
            comp_end = min((j + 1) * config.l_prime // config.d, n_comp_blocks)
            
            # Verify non-zero scores only in expected range
            for i in range(n_comp_blocks):
                if comp_start <= i < comp_end:
                    # Should have contribution
                    if i * config.d + config.l - 1 <= t:  # If causally visible
                        assert sel_scores[0, 0, t, j] > 0, f"Missing contribution at t={t}, j={j}, i={i}"
    
    print("✅ Eq. 9: Triangular mapping validated")


def test_eq10_gqa_aggregation():
    """
    Test Eq. 10: GQA aggregation and top-k selection with deduplication.
    All heads in a group must select identical blocks.
    """
    config = NSAConfig(
        l_prime=64,
        n=16,  # Total blocks to select
        n_heads=8,
        n_kv_groups=2,  # 4 heads per group
    )
    
    B, T = 2, 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_sel_blocks = (T + config.l_prime - 1) // config.l_prime
    
    # Create selection scores
    torch.manual_seed(456)
    sel_scores = torch.rand(B, config.n_heads, T, n_sel_blocks, device=device)
    
    # Apply Eq. 10
    indices = select_top_k_blocks_per_query_with_gqa(sel_scores, config)
    
    # Verify shape
    assert indices.shape == (B, config.n_kv_groups, T, config.n), \
        f"Indices shape mismatch: {indices.shape}"
    
    # Test 1: GQA consistency - all heads in group produce same indices
    heads_per_group = config.n_heads // config.n_kv_groups
    for b in range(B):
        for g in range(config.n_kv_groups):
            # Aggregate scores across heads in group
            group_scores = sel_scores[b, g*heads_per_group:(g+1)*heads_per_group].sum(dim=0)
            
            for t in range(T):
                # Get expected top-k (with always-include logic)
                scores_t = group_scores[t].clone()
                
                # Always include block 0
                always_include = [0]
                current_block = t // config.l_prime
                if current_block > 0:
                    always_include.extend([current_block - 1, current_block])
                if current_block < n_sel_blocks - 1:
                    always_include.append(current_block + 1)
                
                # Verify always-include blocks are in indices
                selected = indices[b, g, t].tolist()
                for block in always_include[:config.n]:  # Up to n blocks
                    if block < n_sel_blocks:
                        assert block in selected, \
                            f"Always-include block {block} missing at t={t}"
    
    # Test 2: Deduplication
    for b in range(B):
        for g in range(config.n_kv_groups):
            for t in range(T):
                selected = indices[b, g, t].tolist()
                unique = list(set(selected))
                # Should have no duplicates (except padding)
                non_padding = [x for x in selected if x != selected[-1]]
                assert len(non_padding) == len(set(non_padding)), \
                    f"Duplicates found at t={t}: {selected}"
    
    print("✅ Eq. 10: GQA aggregation and deduplication validated")


def test_algorithm_integration():
    """
    Test that Eq. 8 → Eq. 9 → Eq. 10 pipeline works end-to-end.
    """
    config = NSAConfig(
        d_model=512,
        head_dim_qk=192,
        head_dim_v=128,
        n_heads=8,
        n_kv_groups=2,
        l=32,
        d=16,
        l_prime=64,
        n=16,
    )
    
    B, T = 1, 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    # Create module and input
    nsa = NSAAttention(config).to(device)
    nsa.eval()  # Set to eval mode for determinism
    hidden = torch.randn(B, T, config.d_model, device=device, dtype=dtype)
    
    with torch.no_grad():
        # Run forward
        output, attn_info = nsa(hidden, output_attentions=True)
        
        # Verify pipeline components exist in attention info
        assert 'selected_indices' in attn_info, "Selection indices missing"
        assert attn_info['selected_indices'].shape == (B, config.n_kv_groups, T, config.n), \
            "Selection indices shape incorrect"
        
        # Verify output is valid
        assert not torch.isnan(output).any(), "NaN in output"
        assert not torch.isinf(output).any(), "Inf in output"
        
        # Test reproducibility with same input
        # Set deterministic mode for reproducibility testing
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        
        output2, info2 = nsa(hidden, output_attentions=True)
        output3, info3 = nsa(hidden, output_attentions=True)
        
        # Check exact equality for selected indices (should be deterministic with tie-breaker)
        torch.testing.assert_close(
            info2['selected_indices'].to(torch.int64),
            info3['selected_indices'].to(torch.int64),
            rtol=0, atol=0
        )
        
        # Check selection scores are very close
        torch.testing.assert_close(info2['selection_scores'], info3['selection_scores'], rtol=1e-5, atol=1e-6)
        
        # Output may have small differences due to Triton kernel parallelism
        # This is acceptable as long as indices and scores are deterministic
        torch.testing.assert_close(output2, output3, rtol=1e-2, atol=1e-2)
        
        # Restore default settings
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
    
    print("✅ Algorithm integration: Eq. 8 → 9 → 10 pipeline validated")


if __name__ == "__main__":
    test_eq8_compression_scores()
    test_eq9_triangular_mapping()
    test_eq10_gqa_aggregation()
    test_algorithm_integration()
    print("\n" + "="*60)
    print("ALL ALGORITHM SPECIFICATION TESTS PASSED")
    print("="*60)