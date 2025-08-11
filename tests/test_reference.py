#!/usr/bin/env python3
"""
Test the reference implementations to ensure they work correctly.
"""

import torch
import triton
import math
import sys

sys.path.append(".")

from nsa.reference import (
    sliding_window_attention,
    compression_attention,
    compress_kv_simple,
    compare_outputs,
    generate_test_tensors,
)

from nsa import (
    _nsa_sliding_window_fwd_kernel,
    _nsa_compression_fwd_kernel,
)


def test_sliding_window_reference():
    """Test sliding window reference implementation."""
    print("=" * 60)
    print("TEST 1: Sliding Window Reference")
    print("=" * 60)

    # Generate test tensors
    B, H, G, T = 2, 8, 2, 64
    dk = dv = 32
    window_size = 16

    q, k, v = generate_test_tensors(B, H, G, T, dk, dv)

    print("Input shapes:")
    print(f"  Q: {q.shape} (B={B}, H={H}, T={T}, dk={dk})")
    print(f"  K: {k.shape} (B={B}, G={G}, dk={dk}, T={T}) <- NOTE: dk before T!")
    print(f"  V: {v.shape} (B={B}, G={G}, T={T}, dv={dv})")
    print(f"  Window size: {window_size}")

    # Run reference
    try:
        with torch.no_grad():
            output = sliding_window_attention(q, k, v, window_size)
        print(f"Output shape: {output.shape}")
        print(f"Output norm: {output.norm():.4f}")
        print("✅ Sliding window reference works!")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_compression_reference():
    """Test compression reference implementation."""
    print("\n" + "=" * 60)
    print("TEST 2: Compression Reference")
    print("=" * 60)

    B, H, G, T = 2, 8, 2, 64
    dk = dv = 32
    block_size = 16
    stride = 8

    q, k, v = generate_test_tensors(B, H, G, T, dk, dv)

    # Compress K and V
    k_compressed, v_compressed, block_ends = compress_kv_simple(
        k, v, block_size, stride
    )

    print("Compression setup:")
    print(f"  Block size: {block_size}, Stride: {stride}")
    print(f"  K compressed: {k_compressed.shape}")
    print(f"  V compressed: {v_compressed.shape}")
    print(f"  Block ends: {block_ends}")

    # Run reference
    try:
        with torch.no_grad():
            output = compression_attention(q, k_compressed, v_compressed, block_ends)
        print(f"Output shape: {output.shape}")
        print(f"Output norm: {output.norm():.4f}")
        print("✅ Compression reference works!")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_against_triton_kernels():
    """Test reference implementations against Triton kernels."""
    print("\n" + "=" * 60)
    print("TEST 3: Reference vs Triton Kernels")
    print("=" * 60)

    # Test configuration
    B, H, G, T = 2, 8, 2, 64
    dk = dv = 32
    window_size = 16

    q, k, v = generate_test_tensors(B, H, G, T, dk, dv)
    scale = 1.0 / math.sqrt(dk)

    print("Testing sliding window...")

    # Reference
    with torch.no_grad():
        out_ref = sliding_window_attention(q, k, v, window_size, scale)

    # Triton kernel
    out_kernel = torch.zeros_like(out_ref)
    L = torch.zeros(B, H, T, device="cuda", dtype=torch.float32)
    M = torch.zeros(B, H, T, device="cuda", dtype=torch.float32)

    grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)

    _nsa_sliding_window_fwd_kernel[grid](
        q,
        k,
        v,
        out_kernel,
        scale,
        L,
        M,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(3),
        k.stride(2),  # CRITICAL: swap strides!
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out_kernel.stride(0),
        out_kernel.stride(1),
        out_kernel.stride(2),
        out_kernel.stride(3),
        L.stride(0),
        L.stride(1),
        L.stride(2),
        M.stride(0),
        M.stride(1),
        M.stride(2),
        Z=B,
        H=H,
        N_KV_GROUPS=G,
        N_CTX=T,
        WINDOW_SIZE=window_size,
        HEAD_DIM_QK=dk,
        HEAD_DIM_V=dv,
        BLOCK_M=32,
        BLOCK_N=32,
    )

    # Compare
    passed, rel_error = compare_outputs(
        out_kernel, out_ref, tolerance=1e-3, name="Sliding Window"
    )

    if passed:
        print(f"✅ Sliding window matches! Relative error: {rel_error:.6f}")
    else:
        print(f"❌ Sliding window mismatch! Relative error: {rel_error:.6f}")

    # Test compression
    print("\nTesting compression...")

    block_size = 16
    stride = 8
    k_compressed, v_compressed, block_ends = compress_kv_simple(
        k, v, block_size, stride
    )

    # Reference
    with torch.no_grad():
        out_ref = compression_attention(
            q, k_compressed, v_compressed, block_ends, scale
        )

    # Triton kernel
    out_kernel = torch.zeros_like(out_ref)
    L = torch.zeros(B, H, T, device="cuda", dtype=torch.float32)
    M = torch.zeros(B, H, T, device="cuda", dtype=torch.float32)

    _nsa_compression_fwd_kernel[grid](
        q,
        k_compressed,
        v_compressed,
        block_ends,
        out_kernel,
        scale,
        L,
        M,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_compressed.stride(0),
        k_compressed.stride(1),
        k_compressed.stride(3),
        k_compressed.stride(2),
        v_compressed.stride(0),
        v_compressed.stride(1),
        v_compressed.stride(2),
        v_compressed.stride(3),
        out_kernel.stride(0),
        out_kernel.stride(1),
        out_kernel.stride(2),
        out_kernel.stride(3),
        L.stride(0),
        L.stride(1),
        L.stride(2),
        M.stride(0),
        M.stride(1),
        M.stride(2),
        Z=B,
        H=H,
        N_KV_GROUPS=G,
        N_CTX_Q=T,
        N_BLOCKS=len(block_ends),
        HEAD_DIM_QK=dk,
        HEAD_DIM_V=dv,
        BLOCK_M=32,
        BLOCK_N=32,
    )

    # Compare
    passed2, rel_error2 = compare_outputs(
        out_kernel, out_ref, tolerance=1e-3, name="Compression"
    )

    if passed2:
        print(f"✅ Compression matches! Relative error: {rel_error2:.6f}")
    else:
        print(f"❌ Compression mismatch! Relative error: {rel_error2:.6f}")

    return passed and passed2


def test_gradients():
    """Test that gradients flow correctly through reference implementations."""
    print("\n" + "=" * 60)
    print("TEST 4: Gradient Flow")
    print("=" * 60)

    B, H, G, T = 1, 4, 2, 32
    dk = dv = 16
    window_size = 8

    q, k, v = generate_test_tensors(B, H, G, T, dk, dv)

    # Sliding window with gradients
    output = sliding_window_attention(q, k, v, window_size)
    loss = output.sum()
    loss.backward()

    print("Sliding window gradients:")
    print(f"  dQ norm: {q.grad.norm():.4f}")
    print(f"  dK norm: {k.grad.norm():.4f}")
    print(f"  dV norm: {v.grad.norm():.4f}")

    # Reset gradients
    q.grad = None
    k.grad = None
    v.grad = None

    # Compression with gradients
    k_compressed, v_compressed, block_ends = compress_kv_simple(
        k, v, block_size=8, stride=4
    )
    k_compressed = k_compressed.detach().requires_grad_(True)
    v_compressed = v_compressed.detach().requires_grad_(True)

    output = compression_attention(q, k_compressed, v_compressed, block_ends)
    loss = output.sum()
    loss.backward()

    print("\nCompression gradients:")
    print(f"  dQ norm: {q.grad.norm():.4f}")
    print(f"  dK_compressed norm: {k_compressed.grad.norm():.4f}")
    print(f"  dV_compressed norm: {v_compressed.grad.norm():.4f}")

    print("✅ Gradients flow correctly!")
    return True


def test_causality():
    """Test that attention is properly causal."""
    print("\n" + "=" * 60)
    print("TEST 5: Causality Check")
    print("=" * 60)

    B, H, G, T = 1, 2, 1, 8
    dk = dv = 16
    window_size = 100  # Large window to test causality

    # Create two inputs that differ in future positions
    torch.manual_seed(42)
    q1, k1, v1 = generate_test_tensors(B, H, G, T, dk, dv)

    q2 = q1.clone()
    k2 = k1.clone()
    v2 = v1.clone()

    # Change future positions
    q2[:, :, T // 2 :, :] = torch.randn_like(q2[:, :, T // 2 :, :]) * 0.02
    k2[:, :, :, T // 2 :] = torch.randn_like(k2[:, :, :, T // 2 :]) * 0.02
    v2[:, :, T // 2 :, :] = torch.randn_like(v2[:, :, T // 2 :, :]) * 0.02

    with torch.no_grad():
        out1 = sliding_window_attention(q1, k1, v1, window_size)
        out2 = sliding_window_attention(q2, k2, v2, window_size)

    # First half should be identical
    first_half_diff = (out1[:, :, : T // 2] - out2[:, :, : T // 2]).abs().max()
    second_half_diff = (out1[:, :, T // 2 :] - out2[:, :, T // 2 :]).abs().max()

    print(f"Max difference in first half: {first_half_diff:.6f}")
    print(f"Max difference in second half: {second_half_diff:.6f}")

    if first_half_diff < 1e-6:
        print("✅ Causal masking correct (first half identical)")
        return True
    else:
        print("❌ Causal masking broken!")
        return False


if __name__ == "__main__":
    print("TESTING NSA REFERENCE IMPLEMENTATIONS")
    print("=" * 60)

    tests = [
        ("Sliding Window", test_sliding_window_reference),
        ("Compression", test_compression_reference),
        ("Reference vs Triton", test_against_triton_kernels),
        ("Gradient Flow", test_gradients),
        ("Causality", test_causality),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test {name} crashed: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
        all_passed = all_passed and passed

    if all_passed:
        print("\n🎉 ALL REFERENCE TESTS PASSED!")
        print("Reference implementations are correct and match Triton kernels!")
    else:
        print("\n❌ Some tests failed.")

    sys.exit(0 if all_passed else 1)
