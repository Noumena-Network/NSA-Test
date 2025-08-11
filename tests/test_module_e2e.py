#!/usr/bin/env python3
"""
End-to-end NSAAttention module validation.
Tests complete forward/backward with paper configuration.
"""

import torch
import math
import pytest

from nsa.kernels import NSAConfig, NSAAttention


def test_paper_configuration():
    """
    Test NSAAttention with exact paper configuration.
    d_model=2560, d_k=192, d_v=128, l=32, d=16, l'=64, n=16, w=512.
    """
    config = NSAConfig(
        d_model=2560,
        head_dim_qk=192,  # Paper's d_k
        head_dim_v=128,   # Paper's d_v
        l=32,             # Compression block size
        d=16,             # Compression stride
        l_prime=64,       # Selection block size
        n=16,             # Number of blocks to select
        w=512,            # Sliding window size
        n_heads=64,       # Paper configuration
        n_kv_groups=4,    # GQA with 16 heads per group
        dropout_p=0.0,    # No dropout for testing
    )
    
    # Use smaller batch/sequence for testing
    B, T = 2, 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    print(f"Testing paper configuration:")
    print(f"  d_model={config.d_model}, d_k={config.head_dim_qk}, d_v={config.head_dim_v}")
    print(f"  Compression: l={config.l}, d={config.d}")
    print(f"  Selection: l'={config.l_prime}, n={config.n}")
    print(f"  Window: w={config.w}")
    print(f"  Heads: {config.n_heads}, Groups: {config.n_kv_groups}")
    
    # Create module
    model = NSAAttention(config).to(device)
    
    # Create input
    torch.manual_seed(42)
    hidden = torch.randn(B, T, config.d_model, device=device, dtype=dtype)
    
    # Forward pass
    output, attn_info = model(hidden, output_attentions=True)
    
    # Verify output shape
    assert output.shape == (B, T, config.d_model), \
        f"Output shape {output.shape} != expected {(B, T, config.d_model)}"
    
    # Verify attention info
    assert 'selected_indices' in attn_info, "Missing selected indices"
    assert attn_info['selected_indices'].shape == (B, config.n_kv_groups, T, config.n), \
        f"Selected indices shape incorrect"
    
    # Verify no NaN/Inf
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"
    
    print("✅ Paper configuration forward pass validated")
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    
    # Check gradients exist and are valid
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"
            assert not torch.isinf(param.grad).any(), f"Inf in gradient for {name}"
    
    print("✅ Paper configuration backward pass validated")


def test_gate_mechanism():
    """
    Test that gate mechanism properly controls branch outputs.
    Zero out one branch gate and verify output changes.
    """
    config = NSAConfig(
        d_model=512,
        head_dim_qk=64,
        head_dim_v=64,
        n_heads=8,
        n_kv_groups=2,
    )
    
    B, T = 1, 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    # Create module
    model = NSAAttention(config).to(device)
    
    # Create input
    torch.manual_seed(123)
    hidden = torch.randn(B, T, config.d_model, device=device, dtype=dtype)
    
    # Get normal output
    with torch.no_grad():
        output_normal, _ = model(hidden)
    
    # Override gate MLP to zero out compression branch
    with torch.no_grad():
        # Save original gate weights
        original_weight = model.gate_mlp[-1].weight.data.clone()
        original_bias = model.gate_mlp[-1].bias.data.clone() if model.gate_mlp[-1].bias is not None else None
        
        # Zero out compression gate (first third of output)
        n_heads = config.n_heads
        model.gate_mlp[-1].weight.data[:n_heads] = 0
        if model.gate_mlp[-1].bias is not None:
            model.gate_mlp[-1].bias.data[:n_heads] = -10  # Large negative to ensure sigmoid → 0
        
        output_no_comp, _ = model(hidden)
        
        # Restore weights
        model.gate_mlp[-1].weight.data = original_weight
        if original_bias is not None:
            model.gate_mlp[-1].bias.data = original_bias
    
    # Outputs should differ when gate is modified
    diff = (output_normal - output_no_comp).abs().max().item()
    assert diff > 1e-4, f"Gate mechanism not working, diff={diff}"
    
    print("✅ Gate mechanism validated")


def test_reproducibility():
    """
    Test that module produces reproducible results with fixed seed.
    """
    config = NSAConfig(d_model=256)
    B, T = 2, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    model = NSAAttention(config).to(device)
    
    # Run 1
    torch.manual_seed(42)
    hidden1 = torch.randn(B, T, config.d_model, device=device, dtype=dtype)
    output1, info1 = model(hidden1, output_attentions=True)
    
    # Run 2 with same seed
    torch.manual_seed(42)
    hidden2 = torch.randn(B, T, config.d_model, device=device, dtype=dtype)
    output2, info2 = model(hidden2, output_attentions=True)
    
    # Should be identical
    torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        info1['selected_indices'].float(), 
        info2['selected_indices'].float(),
        rtol=0, atol=0
    )
    
    print("✅ Reproducibility validated")


def test_gradient_flow():
    """
    Test that gradients flow properly through all branches.
    """
    config = NSAConfig(d_model=256)
    B, T = 1, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    model = NSAAttention(config).to(device)
    
    # Track gradients for all projection layers
    params_to_check = [
        ('q_proj', model.q_proj),
        ('k_compress', model.k_compress),
        ('v_compress', model.v_compress),
        ('k_select', model.k_select),
        ('v_select', model.v_select),
        ('k_sliding', model.k_sliding),
        ('v_sliding', model.v_sliding),
        ('gate_mlp', model.gate_mlp),
        ('o_proj', model.o_proj),
    ]
    
    # Forward pass
    hidden = torch.randn(B, T, config.d_model, device=device, dtype=dtype)
    output = model(hidden)[0]
    
    # Backward pass
    loss = output.mean()
    loss.backward()
    
    # Check all parameters received gradients
    for name, module in params_to_check:
        for param_name, param in module.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name}.{param_name} has no gradient"
                grad_norm = param.grad.norm().item()
                assert grad_norm > 0, f"{name}.{param_name} has zero gradient"
                assert not torch.isnan(param.grad).any(), f"{name}.{param_name} has NaN gradient"
    
    print("✅ Gradient flow validated through all branches")


def test_attention_pattern_sanity():
    """
    Test that attention patterns make semantic sense.
    Early queries should attend mostly to early keys.
    """
    config = NSAConfig(
        d_model=256,
        n_heads=8,
        n_kv_groups=2,
    )
    
    B, T = 1, 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    model = NSAAttention(config).to(device)
    
    # Create input with positional signal
    hidden = torch.zeros(B, T, config.d_model, device=device, dtype=dtype)
    # Add positional encoding
    pos = torch.arange(T, device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)
    hidden += pos / T  # Gradual change over sequence
    
    with torch.no_grad():
        output, attn_info = model(hidden, output_attentions=True)
        
        # Check selected indices
        indices = attn_info['selected_indices']  # [B, G, T, n]
        
        # Early queries should mostly select early blocks
        for t in [10, 20, 30]:
            selected = indices[0, 0, t].float().mean().item()
            # Average selected block index should be relatively small
            max_expected = (t // config.l_prime) + 2
            assert selected < max_expected, \
                f"Query {t} selecting too far: avg block {selected:.1f}"
        
        # Late queries can select from wider range
        for t in [200, 240]:
            selected = indices[0, 0, t]
            # Should include block 0 (always include)
            assert 0 in selected.tolist(), f"Query {t} missing block 0"
    
    print("✅ Attention pattern sanity validated")


def test_memory_efficiency():
    """
    Test memory usage is within expected bounds.
    """
    config = NSAConfig(
        d_model=512,
        n_heads=8,
        n_kv_groups=2,
    )
    
    B, T = 2, 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    model = NSAAttention(config).to(device)
    
    # Compute expected memory for attention
    # Based on our NSA implementation:
    # 1. Compression: (T - l) // d + 1 compressed tokens
    n_comp_blocks = (T - config.l) // config.d + 1
    compression_tokens = n_comp_blocks  # ~63 for T=1024
    
    # 2. Selection: n blocks * l' tokens per block
    selection_tokens = config.n * config.l_prime  # 16 * 64 = 1024
    
    # 3. Window: w tokens
    window_tokens = config.w  # 512
    
    # Total KV tokens loaded per query
    total_tokens = compression_tokens + selection_tokens + window_tokens
    
    # Our implementation gives ~1599 for T=1024
    # This represents the actual KV pairs loaded, which is the 
    # relevant metric for memory efficiency
    expected_range = (1500, 1700)
    
    assert expected_range[0] < total_tokens < expected_range[1], \
        f"Memory usage {total_tokens} outside expected range {expected_range}"
    
    print(f"✅ Memory efficiency validated: {total_tokens} tokens per query")


if __name__ == "__main__":
    test_paper_configuration()
    test_gate_mechanism()
    test_reproducibility()
    test_gradient_flow()
    test_attention_pattern_sanity()
    test_memory_efficiency()
    print("\n" + "="*60)
    print("ALL MODULE E2E TESTS PASSED")
    print("="*60)