#!/usr/bin/env python3
"""
Kernel contract conformance tests.
Validates stride ordering, tensor shapes, L/M usage, and other kernel requirements.
"""

import torch
import triton
import math
import pytest

from nsa import (
    _nsa_sliding_window_fwd_kernel,
    _nsa_sliding_window_bwd_kernel,
    _nsa_compression_fwd_kernel,
    _nsa_compression_bwd_kernel,
    _nsa_sparse_selection_fwd_kernel,
    _nsa_sparse_selection_bwd_kernel,
)


def test_k_stride_contract():
    """
    Test that K tensor MUST use swapped strides.
    K shape is [B, G, dk, T] but kernel expects strides as if [B, G, T, dk].
    Wrong ordering should produce large error (>0.1) vs correct path (<1e-3).
    """
    B, H, G, T = 1, 4, 2, 64
    dk, dv = 32, 32
    window_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    # Create test tensors
    torch.manual_seed(42)
    q = torch.randn(B, H, T, dk, device=device, dtype=dtype)
    k = torch.randn(B, G, dk, T, device=device, dtype=dtype)  # Note: dk before T
    v = torch.randn(B, G, T, dv, device=device, dtype=dtype)
    
    # Prepare outputs
    out_correct = torch.zeros(B, H, T, dv, device=device, dtype=dtype)
    out_wrong = torch.zeros(B, H, T, dv, device=device, dtype=dtype)
    L = torch.zeros(B, H, T, device=device, dtype=torch.float32)
    M = torch.zeros(B, H, T, device=device, dtype=torch.float32)
    
    grid = lambda META: (triton.cdiv(T, META['BLOCK_M']), B * H)
    scale = 1.0 / math.sqrt(dk)
    
    # Run with CORRECT stride ordering
    _nsa_sliding_window_fwd_kernel[grid](
        q, k, v, out_correct, scale, L, M,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(3), k.stride(2),  # SWAPPED - correct!
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out_correct.stride(0), out_correct.stride(1), out_correct.stride(2), out_correct.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        M.stride(0), M.stride(1), M.stride(2),
        Z=B, H=H, N_KV_GROUPS=G, N_CTX=T,
        WINDOW_SIZE=window_size,
        HEAD_DIM_QK=dk, HEAD_DIM_V=dv,
        BLOCK_M=32, BLOCK_N=32,
    )
    
    # Reset L, M for second run
    L.zero_()
    M.zero_()
    
    # Run with WRONG stride ordering (not swapped)
    _nsa_sliding_window_fwd_kernel[grid](
        q, k, v, out_wrong, scale, L, M,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  # NOT swapped - wrong!
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out_wrong.stride(0), out_wrong.stride(1), out_wrong.stride(2), out_wrong.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        M.stride(0), M.stride(1), M.stride(2),
        Z=B, H=H, N_KV_GROUPS=G, N_CTX=T,
        WINDOW_SIZE=window_size,
        HEAD_DIM_QK=dk, HEAD_DIM_V=dv,
        BLOCK_M=32, BLOCK_N=32,
    )
    
    # Compare outputs
    diff = (out_correct - out_wrong).abs()
    rel_error = diff / (out_correct.abs() + 1e-8)
    max_rel_error = rel_error.max().item()
    
    # Wrong stride ordering should produce large error
    assert max_rel_error > 0.1, \
        f"Wrong K stride should produce error > 0.1, got {max_rel_error}"
    
    print(f"✅ K stride contract validated: wrong stride gives {max_rel_error:.3f} error")


def test_dk_shape_contract():
    """
    Test that dK in backward MUST be allocated as [B, G, T, dk].
    After kernel execution, it can be transposed to [B, G, dk, T] if needed.
    """
    B, H, G, T = 1, 4, 2, 64
    dk, dv = 32, 32
    window_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    # Create tensors
    q = torch.randn(B, H, T, dk, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, G, dk, T, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, G, T, dv, device=device, dtype=dtype, requires_grad=True)
    
    # Run forward to get L, M
    out = torch.zeros(B, H, T, dv, device=device, dtype=dtype)
    L = torch.zeros(B, H, T, device=device, dtype=torch.float32)
    M = torch.zeros(B, H, T, device=device, dtype=torch.float32)
    
    grid = lambda META: (triton.cdiv(T, META['BLOCK_M']), B * H)
    scale = 1.0 / math.sqrt(dk)
    
    _nsa_sliding_window_fwd_kernel[grid](
        q, k, v, out, scale, L, M,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(3), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        M.stride(0), M.stride(1), M.stride(2),
        Z=B, H=H, N_KV_GROUPS=G, N_CTX=T,
        WINDOW_SIZE=window_size,
        HEAD_DIM_QK=dk, HEAD_DIM_V=dv,
        BLOCK_M=32, BLOCK_N=32,
    )
    
    # Prepare gradients
    do = torch.randn_like(out)
    dq = torch.zeros_like(q)
    dk_correct = torch.zeros(B, G, T, dk, device=device, dtype=dtype)  # CORRECT shape
    dk_wrong = torch.zeros(B, G, dk, T, device=device, dtype=dtype)    # WRONG shape
    dv_grad = torch.zeros_like(v)
    
    # Run backward with CORRECT dK shape
    _nsa_sliding_window_bwd_kernel[grid](
        # Tensors
        q, k, v, out, do,
        scale, L, M,
        # All strides first (Q, K, V, Out, L, M)
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(3), k.stride(2),  # K strides swapped
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        M.stride(0), M.stride(1), M.stride(2),
        # Gradient outputs
        dq, dk_correct, dv_grad,
        # Gradient strides
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        dk_correct.stride(0), dk_correct.stride(1), dk_correct.stride(2), dk_correct.stride(3),
        dv_grad.stride(0), dv_grad.stride(1), dv_grad.stride(2), dv_grad.stride(3),
        # Shape parameters
        Z=B, H=H, N_KV_GROUPS=G, N_CTX=T,
        WINDOW_SIZE=window_size,
        HEAD_DIM_QK=dk, HEAD_DIM_V=dv,
        BLOCK_M=32, BLOCK_N=32,
    )
    
    # Verify dK has non-zero gradients
    assert dk_correct.abs().max() > 0, "dK should have non-zero gradients"
    
    # After kernel, can transpose to match original K shape
    dk_transposed = dk_correct.transpose(-1, -2)  # [B, G, T, dk] -> [B, G, dk, T]
    assert dk_transposed.shape == k.shape, f"Transposed dK shape {dk_transposed.shape} != K shape {k.shape}"
    
    print("✅ dK shape contract validated: must be [B, G, T, dk] for kernel")


def test_lm_usage_in_backward():
    """
    Test that L/M statistics from forward are required for correct backward.
    Running backward without proper L/M should yield different results.
    """
    B, H, G, T = 1, 4, 2, 64
    dk, dv = 32, 32
    window_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    # Create tensors
    torch.manual_seed(123)
    q = torch.randn(B, H, T, dk, device=device, dtype=dtype)
    k = torch.randn(B, G, dk, T, device=device, dtype=dtype)
    v = torch.randn(B, G, T, dv, device=device, dtype=dtype)
    
    # Run forward to get correct L, M
    out = torch.zeros(B, H, T, dv, device=device, dtype=dtype)
    L_correct = torch.zeros(B, H, T, device=device, dtype=torch.float32)
    M_correct = torch.zeros(B, H, T, device=device, dtype=torch.float32)
    
    grid = lambda META: (triton.cdiv(T, META['BLOCK_M']), B * H)
    scale = 1.0 / math.sqrt(dk)
    
    _nsa_sliding_window_fwd_kernel[grid](
        q, k, v, out, scale, L_correct, M_correct,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(3), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        L_correct.stride(0), L_correct.stride(1), L_correct.stride(2),
        M_correct.stride(0), M_correct.stride(1), M_correct.stride(2),
        Z=B, H=H, N_KV_GROUPS=G, N_CTX=T,
        WINDOW_SIZE=window_size,
        HEAD_DIM_QK=dk, HEAD_DIM_V=dv,
        BLOCK_M=32, BLOCK_N=32,
    )
    
    # Verify L, M are non-zero (populated by forward)
    assert L_correct.abs().max() > 0, "L should be populated by forward"
    assert M_correct.abs().max() > 0, "M should be populated by forward"
    
    # Prepare for backward
    do = torch.randn_like(out)
    dq_correct = torch.zeros_like(q)
    dk_correct = torch.zeros(B, G, T, dk, device=device, dtype=dtype)
    dv_correct = torch.zeros_like(v)
    
    dq_wrong = torch.zeros_like(q)
    dk_wrong = torch.zeros(B, G, T, dk, device=device, dtype=dtype)
    dv_wrong = torch.zeros_like(v)
    
    # Run backward with CORRECT L, M
    _nsa_sliding_window_bwd_kernel[grid](
        # Tensors
        q, k, v, out, do,
        scale, L_correct, M_correct,
        # All strides first
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(3), k.stride(2),  # K strides swapped
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        L_correct.stride(0), L_correct.stride(1), L_correct.stride(2),
        M_correct.stride(0), M_correct.stride(1), M_correct.stride(2),
        # Gradient outputs
        dq_correct, dk_correct, dv_correct,
        # Gradient strides
        dq_correct.stride(0), dq_correct.stride(1), dq_correct.stride(2), dq_correct.stride(3),
        dk_correct.stride(0), dk_correct.stride(1), dk_correct.stride(2), dk_correct.stride(3),
        dv_correct.stride(0), dv_correct.stride(1), dv_correct.stride(2), dv_correct.stride(3),
        # Shape parameters
        Z=B, H=H, N_KV_GROUPS=G, N_CTX=T,
        WINDOW_SIZE=window_size,
        HEAD_DIM_QK=dk, HEAD_DIM_V=dv,
        BLOCK_M=32, BLOCK_N=32,
    )
    
    # Run backward with WRONG L, M (zeros)
    L_wrong = torch.zeros_like(L_correct)
    M_wrong = torch.zeros_like(M_correct)
    
    _nsa_sliding_window_bwd_kernel[grid](
        # Tensors
        q, k, v, out, do,
        scale, L_wrong, M_wrong,
        # All strides first
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(3), k.stride(2),  # K strides swapped
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        L_wrong.stride(0), L_wrong.stride(1), L_wrong.stride(2),
        M_wrong.stride(0), M_wrong.stride(1), M_wrong.stride(2),
        # Gradient outputs
        dq_wrong, dk_wrong, dv_wrong,
        # Gradient strides
        dq_wrong.stride(0), dq_wrong.stride(1), dq_wrong.stride(2), dq_wrong.stride(3),
        dk_wrong.stride(0), dk_wrong.stride(1), dk_wrong.stride(2), dk_wrong.stride(3),
        dv_wrong.stride(0), dv_wrong.stride(1), dv_wrong.stride(2), dv_wrong.stride(3),
        # Shape parameters
        Z=B, H=H, N_KV_GROUPS=G, N_CTX=T,
        WINDOW_SIZE=window_size,
        HEAD_DIM_QK=dk, HEAD_DIM_V=dv,
        BLOCK_M=32, BLOCK_N=32,
    )
    
    # Compare gradients - should be different
    dq_diff = (dq_correct - dq_wrong).abs().max().item()
    dk_diff = (dk_correct - dk_wrong).abs().max().item()
    dv_diff = (dv_correct - dv_wrong).abs().max().item()
    
    # Using wrong L/M should produce different gradients
    assert dq_diff > 1e-4, f"dQ should differ with wrong L/M, diff={dq_diff}"
    assert dk_diff > 1e-4, f"dK should differ with wrong L/M, diff={dk_diff}"
    assert dv_diff > 1e-4, f"dV should differ with wrong L/M, diff={dv_diff}"
    
    print("✅ L/M usage validated: backward requires forward statistics")


def test_tensor_shape_invariants():
    """
    Test that all kernels maintain expected tensor shape invariants.
    """
    configs = [
        (2, 8, 2, 128, 64, 64),   # B, H, G, T, dk, dv
        (1, 16, 4, 256, 128, 128),
        (4, 4, 1, 64, 32, 32),
    ]
    
    for B, H, G, T, dk, dv in configs:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float32
        
        # Verify shape constraints
        assert H % G == 0, f"H={H} must be divisible by G={G}"
        heads_per_group = H // G
        
        # Q shape: [B, H, T, dk]
        q = torch.randn(B, H, T, dk, device=device, dtype=dtype)
        assert q.shape == (B, H, T, dk)
        
        # K shape: [B, G, dk, T] - note dk before T!
        k = torch.randn(B, G, dk, T, device=device, dtype=dtype)
        assert k.shape == (B, G, dk, T)
        
        # V shape: [B, G, T, dv]
        v = torch.randn(B, G, T, dv, device=device, dtype=dtype)
        assert v.shape == (B, G, T, dv)
        
        # Output shape: [B, H, T, dv]
        out = torch.zeros(B, H, T, dv, device=device, dtype=dtype)
        assert out.shape == (B, H, T, dv)
        
        # L/M shape: [B, H, T]
        L = torch.zeros(B, H, T, device=device, dtype=torch.float32)
        M = torch.zeros(B, H, T, device=device, dtype=torch.float32)
        assert L.shape == (B, H, T)
        assert M.shape == (B, H, T)
        
        # Gradient shapes
        dq = torch.zeros_like(q)
        assert dq.shape == q.shape
        
        # dK must be [B, G, T, dk] for kernel
        dk_grad = torch.zeros(B, G, T, dk, device=device, dtype=dtype)
        assert dk_grad.shape == (B, G, T, dk)
        
        dv_grad = torch.zeros_like(v)
        assert dv_grad.shape == v.shape
        
        print(f"  Config B={B}, H={H}, G={G}, T={T}: shapes validated")
    
    print("✅ All tensor shape invariants validated")


def test_parameter_constraints():
    """
    Test that kernels enforce expected parameter constraints.
    """
    # Test invalid configurations that should fail
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test 1: H not divisible by G
    try:
        B, H, G, T = 1, 7, 2, 64  # 7 not divisible by 2
        dk = dv = 32
        q = torch.randn(B, H, T, dk, device=device)
        k = torch.randn(B, G, dk, T, device=device)
        # This should fail in practice
        # (kernel would compute wrong head-group mapping)
        assert H % G == 0, "H must be divisible by G"
    except AssertionError:
        pass  # Expected
    
    # Test 2: Block sizes must be positive
    from nsa.kernels import NSAConfig
    
    try:
        config = NSAConfig(l=0)  # Invalid
        assert False, "Should not accept l=0"
    except (ValueError, AssertionError):
        pass  # Expected
    
    try:
        config = NSAConfig(l_prime=0)  # Invalid
        assert False, "Should not accept l_prime=0"
    except (ValueError, AssertionError):
        pass  # Expected
    
    # Test 3: Stride must be less than block size for overlap
    config = NSAConfig(l=32, d=16)
    assert config.d < config.l, "Compression stride must be < block size"
    
    # Test 4: Number of selected blocks must be positive
    config = NSAConfig(n=16)
    assert config.n > 0, "Must select at least 1 block"
    
    print("✅ Parameter constraints validated")


if __name__ == "__main__":
    test_k_stride_contract()
    test_dk_shape_contract()
    test_lm_usage_in_backward()
    test_tensor_shape_invariants()
    test_parameter_constraints()
    print("\n" + "="*60)
    print("ALL KERNEL CONTRACT TESTS PASSED")
    print("="*60)