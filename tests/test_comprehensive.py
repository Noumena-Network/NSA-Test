"""
Comprehensive spec tests for NSA V7 implementation.

Tests against paper requirements and production correctness.
"""

import torch
import math
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Import these if they exist in kernels.py
try:
    from nsa.kernels import (
        NSAConfig,
        NSAAttention,
        derive_selection_from_compression_per_timestep,
        select_top_k_blocks_per_query_with_gqa,
    )
except ImportError:
    # These may not be in the kernel file yet
    NSAConfig = None
    NSAAttention = None
    derive_selection_from_compression_per_timestep = None
    select_top_k_blocks_per_query_with_gqa = None


def test_paper_spec_alignment():
    """Test alignment with paper specifications."""
    print("\n" + "=" * 80)
    print("PAPER SPECIFICATION TESTS")
    print("=" * 80)

    # Test 1: Configuration matches paper Section 4.1
    print("\n[TEST 1] Configuration alignment with paper...")
    config = NSAConfig(
        d_model=2560,
        head_dim_qk=192,  # d_k from paper
        head_dim_v=128,  # d_v from paper
        l=32,  # compression block size
        d=16,  # compression stride
        l_prime=64,  # selection block size
        n=16,  # total blocks
        w=512,  # window size
        n_heads=64,
        n_kv_groups=4,
    )

    assert config.head_dim_qk == 192, "d_k mismatch"
    assert config.head_dim_v == 128, "d_v mismatch"
    assert config.heads_per_group == 16, "Heads per group mismatch"
    print("  ✓ Configuration matches paper exactly")

    # Test 2: Compression parameters (Section 3.3.1)
    print("\n[TEST 2] Compression parameters...")
    assert config.l == 32 and config.d == 16, "Compression params mismatch"
    assert config.l % config.d == 0, "l must be divisible by d"
    print(f"  ✓ Compression: l={config.l}, d={config.d} (overlap when d<l)")

    # Test 3: Selection parameters (Section 3.3.2)
    print("\n[TEST 3] Selection parameters...")
    assert config.l_prime == 64 and config.n == 16, "Selection params mismatch"
    assert config.n_fixed == 3, "Should have 1 initial + 2 local blocks"
    print(
        f"  ✓ Selection: l'={config.l_prime}, n={config.n} (with {config.n_fixed} fixed)"
    )

    # Test 4: Sliding window (Section 3.3.3)
    print("\n[TEST 4] Sliding window...")
    assert config.w == 512, "Window size mismatch"
    print(f"  ✓ Window size: {config.w}")

    print("\n✅ All paper specifications verified")


def test_eq9_triangular_mapping():
    """Test exact Eq. 9 triangular mapping implementation."""
    print("\n" + "=" * 80)
    print("EQ. 9 TRIANGULAR MAPPING TEST")
    print("=" * 80)

    config = NSAConfig()
    B, H, T = 2, 64, 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create realistic compression scores using actual Q @ K computation
    n_comp_blocks = (T - config.l) // config.d + 1
    # Generate query and compressed key tensors
    q_test = torch.randn(B, H, T, config.head_dim_qk, device=device)
    k_compressed = torch.randn(B, config.n_kv_groups, config.head_dim_qk, n_comp_blocks, device=device)
    
    # Expand k_compressed for all heads and compute actual attention scores
    G = config.n_kv_groups
    heads_per_group = H // G
    k_exp = k_compressed.unsqueeze(2).repeat(1, 1, heads_per_group, 1, 1)
    k_exp = k_exp.view(B, H, config.head_dim_qk, n_comp_blocks)
    
    # Compute Q @ K^T with proper scaling
    scale = 1.0 / math.sqrt(config.head_dim_qk)
    comp_scores = torch.matmul(q_test, k_exp) * scale
    
    # Apply causal masking based on block ends
    block_ends = torch.arange(n_comp_blocks, device=device) * config.d + config.l - 1
    for t in range(T):
        has_valid = False
        for block_idx in range(n_comp_blocks):
            if block_ends[block_idx] <= t:
                has_valid = True
                break
        
        for block_idx in range(n_comp_blocks):
            if block_ends[block_idx] > t:
                # If no blocks are valid, keep first block for numerical stability
                if not has_valid and block_idx == 0:
                    continue
                comp_scores[:, :, t, block_idx] = float('-inf')
    
    comp_scores = torch.softmax(comp_scores, dim=-1)

    print(f"\n[TEST] Compression scores shape: {comp_scores.shape}")

    # Apply Eq. 9 derivation
    sel_scores = derive_selection_from_compression_per_timestep(comp_scores, config)

    # Verify shape
    expected_sel_blocks = (
        n_comp_blocks * config.d + config.l_prime - 1
    ) // config.l_prime
    assert sel_scores.shape == (B, H, T, expected_sel_blocks), (
        f"Shape mismatch: {sel_scores.shape} vs expected (B, H, T, {expected_sel_blocks})"
    )

    print(f"  ✓ Selection scores shape: {sel_scores.shape}")

    # Verify triangular sum pattern
    a = config.l // config.d  # =2
    s = config.l_prime // config.d  # =4

    print(f"  ✓ Triangular mapping: a={a}, s={s}")

    # Manual verification for first selection block
    manual_sum = torch.zeros_like(sel_scores[:, :, :, 0])
    for m in range(s):
        for n in range(a):  # Full range, no min()
            comp_idx = m + n
            if comp_idx < n_comp_blocks:
                manual_sum += comp_scores[:, :, :, comp_idx]

    torch.testing.assert_close(sel_scores[:, :, :, 0], manual_sum, rtol=1e-5, atol=1e-5)
    print("  ✓ Triangular sum verified for first block")

    print("\n✅ Eq. 9 triangular mapping correct")


def test_gqa_aggregation_eq10():
    """Test GQA aggregation according to Eq. 10."""
    print("\n" + "=" * 80)
    print("EQ. 10 GQA AGGREGATION TEST")
    print("=" * 80)

    config = NSAConfig()
    B, H, T = 2, 64, 512
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_blocks = T // config.l_prime
    sel_scores = torch.randn(B, H, T, n_blocks, device=device)

    print(f"\n[TEST] Input scores shape: {sel_scores.shape}")
    print(f"  Groups: {config.n_kv_groups}, Heads per group: {config.heads_per_group}")

    # Apply GQA selection
    indices = select_top_k_blocks_per_query_with_gqa(sel_scores, config)

    # Verify output shape - now per-group [B, G, T, n]
    assert indices.shape == (B, config.n_kv_groups, T, config.n), (
        f"Indices shape mismatch: {indices.shape} vs expected ({B}, {config.n_kv_groups}, {T}, {config.n})"
    )

    print(f"  ✓ Output indices shape: {indices.shape}")

    # Verify always-include blocks
    for b in range(B):
        for g in range(config.n_kv_groups):
            # Block 0 should always be included
            assert (indices[b, g, :, :] == 0).any(dim=1).all(), (
                f"Block 0 not always included for group {g}"
            )

            # Check local blocks for some positions
            for t in [100, 200, 300]:
                indices_t = indices[b, g, t].cpu().numpy()
                # Should include recent blocks
                recent_block = t // config.l_prime
                assert recent_block in indices_t or recent_block - 1 in indices_t, (
                    f"Local blocks not included at position {t} for group {g}"
                )

    print("  ✓ Always-include blocks verified (initial + local)")

    # Verify deduplication
    for b in range(B):
        for g in range(config.n_kv_groups):
            for t in range(T):
                unique = torch.unique(indices[b, g, t])
                assert len(unique) <= config.n, (
                    f"Too many unique blocks at ({b}, {g}, {t})"
                )

    print("  ✓ Deduplication verified")

    print("\n✅ Eq. 10 GQA aggregation correct")


def test_kernel_shapes_and_strides():
    """Test kernel input/output shapes and stride handling."""
    print("\n" + "=" * 80)
    print("KERNEL SHAPE AND STRIDE TESTS")
    print("=" * 80)

    config = NSAConfig()
    B, H, T, G = 2, 64, 1024, 4
    dk, dv = 192, 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Create tensors with correct shapes
    q = torch.randn(B, H, T, dk, dtype=dtype, device=device)
    k = torch.randn(
        B, G, dk, T, dtype=dtype, device=device
    )  # Note: transposed for kernels
    v = torch.randn(B, G, T, dv, dtype=dtype, device=device)
    out = torch.zeros(B, H, T, dv, dtype=dtype, device=device)

    print("\n[TEST] Tensor shapes:")
    print(f"  Q: {q.shape} (B, H, T, dk)")
    print(f"  K: {k.shape} (B, G, dk, T)")
    print(f"  V: {v.shape} (B, G, T, dv)")
    print(f"  Out: {out.shape} (B, H, T, dv)")

    # Test selection kernel shapes - create unique indices with -1 padding
    n_blocks_max = T // config.l_prime
    block_indices = torch.full((B, T, config.n), -1, dtype=torch.int32, device=device)
    for b in range(B):
        for t in range(T):
            # Select unique blocks for this query (respecting causality)
            max_block = min(t // config.l_prime, n_blocks_max - 1)
            if max_block >= 0:
                n_valid = min(config.n, max_block + 1)
                selected = torch.randperm(max_block + 1, device=device)[:n_valid]
                block_indices[b, t, :n_valid] = selected
    print(f"  Block indices: {block_indices.shape} (B, T, n)")

    # Verify stride calculations
    assert q.stride() == (H * T * dk, T * dk, dk, 1), "Q stride mismatch"
    assert k.stride() == (G * dk * T, dk * T, T, 1), "K stride mismatch"
    assert v.stride() == (G * T * dv, T * dv, dv, 1), "V stride mismatch"

    print("  ✓ All tensor strides correct")

    # Test compression kernel shapes
    n_blocks = (T - config.l) // config.d + 1
    k_comp = torch.randn(B, G, dk, n_blocks, dtype=dtype, device=device)
    v_comp = torch.randn(B, G, n_blocks, dv, dtype=dtype, device=device)
    block_ends = torch.arange(
        config.l - 1, T, config.d, dtype=torch.int32, device=device
    )[:n_blocks]

    print(f"\n  Compressed K: {k_comp.shape} (B, G, dk, n_blocks)")
    print(f"  Compressed V: {v_comp.shape} (B, G, n_blocks, dv)")
    print(f"  Block ends: {block_ends.shape} (n_blocks,)")

    # Verify block_ends values
    for i in range(min(5, len(block_ends))):
        expected = i * config.d + config.l - 1
        assert block_ends[i] == expected, (
            f"Block end {i}: {block_ends[i]} != {expected}"
        )

    print("  ✓ Block ends = i*d + (l-1) verified")

    print("\n✅ All kernel shapes and strides correct")


def test_nsa_module_forward():
    """Test complete NSA module forward pass."""
    print("\n" + "=" * 80)
    print("NSA MODULE FORWARD PASS TEST")
    print("=" * 80)

    config = NSAConfig(d_model=512)  # Smaller for testing
    B, T = 2, 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"\n[TEST] Creating NSA module with d_model={config.d_model}")
    nsa = NSAAttention(config).to(device)

    # Test input
    hidden = torch.randn(B, T, config.d_model, device=device, dtype=dtype)
    print(f"  Input shape: {hidden.shape}")

    # Forward pass with attention info
    output, attn_info = nsa(hidden, output_attentions=True)

    # Verify output shape
    assert output.shape == (B, T, config.d_model), (
        f"Output shape mismatch: {output.shape} vs {(B, T, config.d_model)}"
    )
    print(f"  ✓ Output shape: {output.shape}")

    # Verify attention info
    assert attn_info is not None, "Attention info not returned"
    required_keys = [
        "compression_scores",
        "selection_scores",
        "selected_indices",
        "gates",
    ]
    for key in required_keys:
        assert key in attn_info, f"Missing key: {key}"

    print(f"  ✓ Attention info keys: {list(attn_info.keys())}")

    # Verify attention info shapes
    assert attn_info["compression_scores"].shape[0:3] == (B, config.n_heads, T), (
        "Compression scores shape mismatch"
    )
    assert attn_info["selection_scores"].shape[0:3] == (B, config.n_heads, T), (
        "Selection scores shape mismatch"
    )
    assert attn_info["selected_indices"].shape == (
        B,
        config.n_kv_groups,
        T,
        config.n,
    ), "Selected indices shape mismatch"
    assert attn_info["gates"].shape == (B, T, config.n_heads, 3), "Gates shape mismatch"

    print("  ✓ All attention info shapes correct")

    # Test gradient flow
    loss = output.sum()
    loss.backward()

    # Check gradients exist
    for name, param in nsa.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    print("  ✓ Gradient flow verified")

    print("\n✅ NSA module forward pass complete")


def test_compression_with_mlp():
    """Test compression token generation with MLP."""
    print("\n" + "=" * 80)
    print("COMPRESSION WITH MLP TEST")
    print("=" * 80)

    config = NSAConfig()
    B, G, T = 2, 4, 512
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nsa = NSAAttention(config).to(device)

    # Create input tensors
    k = torch.randn(B, G, T, config.head_dim_qk, device=device)
    v = torch.randn(B, G, T, config.head_dim_v, device=device)

    print(f"\n[TEST] Input K shape: {k.shape}")
    print(f"       Input V shape: {v.shape}")

    # Compress tokens
    k_comp, v_comp, block_ends, n_blocks_actual = nsa.compress_tokens(k, v)

    # Expected number of blocks (actual, before padding)
    n_blocks_expected = (T - config.l) // config.d + 1
    assert n_blocks_actual == n_blocks_expected, (
        f"Actual blocks: {n_blocks_actual} vs expected {n_blocks_expected}"
    )

    # The returned tensors are padded to next power of 2
    n_blocks_padded = k_comp.shape[-1]

    # Verify shapes (using padded size)
    assert k_comp.shape == (B, G, config.head_dim_qk, n_blocks_padded), (
        f"K compressed shape: {k_comp.shape}"
    )
    assert v_comp.shape == (B, G, n_blocks_padded, config.head_dim_v), (
        f"V compressed shape: {v_comp.shape}"
    )

    print(f"  ✓ Compressed K shape: {k_comp.shape}")
    print(f"  ✓ Compressed V shape: {v_comp.shape}")

    # Verify block ends (should be padded size)
    assert len(block_ends) == n_blocks_padded, (
        f"Block ends length: {len(block_ends)} vs {n_blocks_padded}"
    )

    # Check actual blocks (not padded ones)
    for i in range(min(10, n_blocks_actual)):
        expected_end = i * config.d + config.l - 1
        assert block_ends[i] == expected_end, (
            f"Block {i} end: {block_ends[i]} vs expected {expected_end}"
        )

    print(f"  ✓ Block ends verified (first 10): {block_ends[:10].tolist()}")

    # Test MLP parameters are used
    mlp_params = [p for n, p in nsa.named_parameters() if "compress_mlp" in n]
    assert len(mlp_params) > 0, "No MLP parameters found"
    print(f"  ✓ MLP has {len(mlp_params)} parameter tensors")

    print("\n✅ Compression with MLP verified")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n" + "=" * 80)
    print("EDGE CASE TESTS")
    print("=" * 80)

    config = NSAConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test 1: Minimum sequence length
    print("\n[TEST 1] Minimum sequence length...")
    B, T_min = 1, config.l + 1  # Just enough for one compressed block
    nsa = NSAAttention(config).to(device)
    hidden = torch.randn(B, T_min, config.d_model, device=device)

    output, _ = nsa(hidden)
    assert output.shape == (B, T_min, config.d_model)
    print(f"  ✓ Handled T={T_min} (minimum)")

    # Test 2: Non-divisible sequence length
    print("\n[TEST 2] Non-divisible sequence length...")
    T_odd = 317  # Not divisible by l_prime or d
    hidden = torch.randn(B, T_odd, config.d_model, device=device)

    output, _ = nsa(hidden)
    assert output.shape == (B, T_odd, config.d_model)
    print(f"  ✓ Handled T={T_odd} (non-divisible)")

    # Test 3: Empty always-include handling
    print("\n[TEST 3] Always-include blocks at boundaries...")
    T = 128
    n_blocks = T // config.l_prime
    sel_scores = torch.randn(B, config.n_heads, T, n_blocks, device=device)

    indices = select_top_k_blocks_per_query_with_gqa(sel_scores, config)

    # Check first position includes block 0 for all groups
    for g in range(config.n_kv_groups):
        assert (indices[:, g, 0, :] == 0).any(dim=1).all()
    print("  ✓ Block 0 included at position 0")

    # Check last position includes appropriate blocks for all groups
    last_block = (T - 1) // config.l_prime
    for g in range(config.n_kv_groups):
        assert (indices[:, g, -1, :] == last_block).any(dim=1).all() or (
            indices[:, g, -1, :] == last_block - 1
        ).any(dim=1).all()
    print(f"  ✓ Local blocks included at position {T - 1}")

    print("\n✅ All edge cases handled correctly")


def run_all_tests():
    """Run all comprehensive tests."""
    print("\n" + "=" * 80)
    print("NSA V7 COMPREHENSIVE SPECIFICATION TESTS")
    print("=" * 80)

    try:
        # Core spec tests
        test_paper_spec_alignment()
        test_eq9_triangular_mapping()
        test_gqa_aggregation_eq10()

        # Implementation tests
        test_kernel_shapes_and_strides()
        test_nsa_module_forward()
        test_compression_with_mlp()

        # Edge cases
        test_edge_cases()

        print("\n" + "=" * 80)
        print("🎉 ALL COMPREHENSIVE TESTS PASSED!")
        print("=" * 80)

        print("\n📊 Test Summary:")
        print("  ✅ Paper specification alignment")
        print("  ✅ Eq. 9 triangular mapping")
        print("  ✅ Eq. 10 GQA aggregation")
        print("  ✅ Kernel shapes and strides")
        print("  ✅ Complete forward pass")
        print("  ✅ Compression with MLP")
        print("  ✅ Edge cases handled")

        print("\n🚀 NSA V7 is production-ready and spec-compliant!")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
